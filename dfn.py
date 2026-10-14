"""Simulator for Discrete Fracture Networks."""

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg, eigs, spsolve

LEFT, RIGHT = (
    -1,
    -2,
)  # Sentinel node ids for points lying on the left/right (fixed-head) boundaries


def _cross(a: np.ndarray, b: np.ndarray):
    """z-component of the cross product of (arrays of) 2D vectors."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class DFNModel:
    """A two-dimensional discrete fracture network is a set of fractures within a bounded (rectangular) two-dimensional domain.
    Each fracture is a line segment defined by its length (a single scalar), midpoint location (a pair of coordinates), orientation (a float value),
    and aperture (a float value). These values should be generated according to the following distributions:
        - Midpoints are chosen so that a global density criterion is met. A starting example is a Poisson point process, i.e. so that
        the number of fractures within a box of area V follows such a process with mean proportional to V.
        - Length L follows a power law distribution with probability density function ~ L^{-α-1}
        where 1 <= α <= 3.
        - Orientation θ clusters around a mean orientation μ, with probability density function ~ exp(k * cos(θ - μ))/(2πI_0(k))
        where I_0 is the modified Bessel function of zero order and k is a constant.
        - Aperture A can follow a log-normal distribution, or be power-law dependent on the length L.

    The distributions according to which these values are generated can be modified further upon data from geological field studies.

    Analysis of a DFN is essentially aimed at calculating total fluid flow from the left boundary of the box to the right boundary,
    assuming a given pressure differential between the two sides. This can be deduced from the higher-information question of calculating
    the hydraulic head at every node (fracture intersection) within the network, which in-turn comes from solving a linear equation of the
    form Ah = b, where A encodes the cluster behavior of the network, b is defined by connections to the boundaries, and h is a vector of
    length equal to the number of nodes and gives the hydraulic head at all nodes.

    The left and right boundaries are held at fixed hydraulic heads; the top and bottom boundaries are impermeable (no flow). Fractures
    extending past the domain are truncated at its edges. Flow along each fracture segment between neighbouring nodes obeys the cubic law,
    Q = (ρ g a^3 / 12 μ) * Δh / ℓ per unit depth, and A, b express conservation of mass at every node.
    """

    def __init__(
        self,
        width: float,  # Width of the bounding box
        height: float,  # Height of the bounding box
        density: float,  # Mean number of fracture midpoints per unit area (intensity of the Poisson point process)
        alpha: float,  # Power law exponent governing the distribution of fracture lengths
        mu: float,  # Mean of orientation distribution function
        k: float,  # Concentration of orientation distribution function (k = 0 is uniform; larger k clusters more tightly about mu)
        *,
        l_min: float = 1.0,  # Lower cutoff of the fracture length distribution
        l_max: float = np.inf,  # Upper cutoff of the fracture length distribution
        aperture_model: str = "lognormal",  # Either "lognormal" or "powerlaw"
        aperture_log_mean: float = np.log(
            1e-4
        ),  # Mean of log(aperture), for the log-normal model
        aperture_log_std: float = 0.5,  # Standard deviation of log(aperture), for the log-normal model
        aperture_coeff: float = 1e-4,  # aperture = aperture_coeff * L**aperture_exponent, for the power-law model
        aperture_exponent: float = 0.5,
        fluid_density: float = 1000.0,  # Density of the fluid, ρ
        fluid_viscosity: float = 1e-3,  # Dynamic viscosity of the fluid, μ
        gravity: float = 9.81,  # Gravitational acceleration, g
        seed: int | None = None,  # Seed for the random number generator
    ):
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if density < 0:
            raise ValueError("density must be non-negative")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if k < 0:
            raise ValueError("k must be non-negative")
        if not 0 < l_min < l_max:
            raise ValueError("require 0 < l_min < l_max")
        if aperture_model not in ("lognormal", "powerlaw"):
            raise ValueError(f"unknown aperture_model {aperture_model!r}")

        self.width = width
        self.height = height
        self.density = density
        self.alpha = alpha
        self.mu = mu
        self.k = k
        self.l_min = l_min
        self.l_max = l_max
        self.aperture_model = aperture_model
        self.aperture_log_mean = aperture_log_mean
        self.aperture_log_std = aperture_log_std
        self.aperture_coeff = aperture_coeff
        self.aperture_exponent = aperture_exponent
        self.fluid_density = fluid_density
        self.fluid_viscosity = fluid_viscosity
        self.gravity = gravity
        self._rng = np.random.default_rng(seed)

        # Populated by cluster_and_prune(), analyze() and solve() respectively
        self.clusters = self.nodes = self.node_fractures = None
        self.A = self.b = None
        self.heads = self.total_flow = None

        self._generate_dfn()

    def _generate_dfn(self):
        """Stochastically generates a DFN instance, storing the results within the class."""
        rng = self._rng

        # Midpoints: homogeneous Poisson point process over the box
        n = rng.poisson(self.density * self.width * self.height)
        self.midpoints = rng.uniform((0.0, 0.0), (self.width, self.height), size=(n, 2))

        # Lengths: (truncated) power law with pdf ~ L^{-α-1} on [l_min, l_max], sampled by inverting the CDF
        lo, hi = self.l_min**-self.alpha, self.l_max**-self.alpha
        self.lengths = (lo - rng.uniform(size=n) * (lo - hi)) ** (-1 / self.alpha)

        # Orientations: von Mises distribution about mu
        self.orientations = rng.vonmises(self.mu, self.k, size=n)

        # Apertures
        if self.aperture_model == "lognormal":
            self.apertures = rng.lognormal(
                self.aperture_log_mean, self.aperture_log_std, size=n
            )
        else:
            self.apertures = self.aperture_coeff * self.lengths**self.aperture_exponent

        self._clip_to_domain()

    def _clip_to_domain(self):
        """Truncates each fracture to the bounding box, storing its endpoints and which of them lie on the left/right boundaries."""
        d = np.column_stack([np.cos(self.orientations), np.sin(self.orientations)])
        # Each fracture is midpoint + s * d; find the range of s that stays inside the box
        s_lo, s_hi = -self.lengths / 2, self.lengths / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            for axis, extent in ((0, self.width), (1, self.height)):
                m, di = self.midpoints[:, axis], d[:, axis]
                s_lo = np.maximum(
                    s_lo,
                    np.where(
                        di > 0, -m / di, np.where(di < 0, (extent - m) / di, -np.inf)
                    ),
                )
                s_hi = np.minimum(
                    s_hi,
                    np.where(
                        di > 0, (extent - m) / di, np.where(di < 0, -m / di, np.inf)
                    ),
                )

        self.clipped_lengths = s_hi - s_lo
        endpoints = (
            self.midpoints[:, None, :]
            + np.stack([s_lo, s_hi], axis=1)[:, :, None] * d[:, None, :]
        )
        self.endpoints = np.clip(
            endpoints, 0.0, (self.width, self.height)
        )  # Shape (n, 2, 2): fracture, start/end, x/y

        tol = 1e-9 * max(self.width, self.height)
        self._on_left = (
            self.endpoints[:, :, 0] <= tol
        )  # Shape (n, 2): whether each endpoint touches the left boundary
        self._on_right = self.endpoints[:, :, 0] >= self.width - tol
        self.endpoints[:, :, 0][self._on_left] = 0.0
        self.endpoints[:, :, 0][self._on_right] = self.width

    def _intersections(self):
        """Finds all pairs of intersecting fractures.

        Returns an (m, 2) array of fracture index pairs (i < j), and an (m, 2) array giving the position of the intersection along
        each fracture, as a fraction of the way from its start point to its end point.
        """
        p = self.endpoints[:, 0]
        r = self.endpoints[:, 1] - p
        pairs, params = [], []
        for i in range(len(p) - 1):
            # Solve p_i + t r_i = p_j + u r_j against every later fracture j at once
            q, rj = p[i + 1 :] - p[i], r[i + 1 :]
            denom = _cross(r[i], rj)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = _cross(q, rj) / denom
                u = _cross(q, r[i]) / denom
            hit = np.flatnonzero(
                (denom != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
            )
            pairs.append(np.column_stack([np.full(len(hit), i), i + 1 + hit]))
            params.append(np.column_stack([t[hit], u[hit]]))

        if not pairs:
            return np.empty((0, 2), dtype=int), np.empty((0, 2))
        return np.concatenate(pairs).astype(int), np.concatenate(params)

    def cluster_and_prune(self):
        """Clusters overlapping fractures, and prunes away any clusters which are isolated (i.e. which do not connect to any boundaries).
        Finally, defines a set of `nodes' which are intersection points between fractures. Stores the named clusters and nodes within the class.

        Only the left and right boundaries count here: a cluster touching just the impermeable top/bottom boundaries has no fixed head
        and would make A singular.
        """
        n = len(self.lengths)
        pairs, params = self._intersections()

        graph = sp.coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)
        )
        n_clusters, labels = connected_components(graph, directed=False)
        on_boundary = (self._on_left | self._on_right).any(axis=1)
        keep = np.bincount(labels, weights=on_boundary, minlength=n_clusters) > 0

        self.labels = labels  # Cluster id of every fracture, including pruned ones
        self.active = keep[
            labels
        ]  # Whether each fracture belongs to a retained cluster
        self.clusters = [np.flatnonzero(labels == c) for c in np.flatnonzero(keep)]

        node_mask = self.active[pairs[:, 0]]
        self.node_fractures = pairs[node_mask]  # The two fractures meeting at each node
        self._node_params = params[node_mask]
        i, t = self.node_fractures[:, 0], self._node_params[:, 0]
        self.nodes = self.endpoints[i, 0] + t[:, None] * (
            self.endpoints[i, 1] - self.endpoints[i, 0]
        )

    def analyze(self, head_left: float = 1.0, head_right: float = 0.0):
        """Analyzes the discrete fracture network instance, generating the matrix A and the vector b as described in the class definition
        (and storing them within the class)."""
        self.cluster_and_prune()
        self.head_left, self.head_right = head_left, head_right
        m = len(self.nodes)

        # Gather every point of interest along each fracture: intersection nodes, plus endpoints on the fixed-head boundaries
        fractures = [self.node_fractures[:, 0], self.node_fractures[:, 1]]
        positions = [self._node_params[:, 0], self._node_params[:, 1]]
        ids = [np.arange(m), np.arange(m)]
        for end in (0, 1):
            for on_boundary, tag in ((self._on_left, LEFT), (self._on_right, RIGHT)):
                f = np.flatnonzero(on_boundary[:, end] & self.active)
                fractures.append(f)
                positions.append(np.full(len(f), float(end)))
                ids.append(np.full(len(f), tag))
        fractures, positions, ids = (
            np.concatenate(x) for x in (fractures, positions, ids)
        )
        order = np.lexsort((positions, fractures))
        fractures, positions, ids = fractures[order], positions[order], ids[order]

        # Consecutive points along the same fracture are joined by a pipe whose conductance follows the cubic law
        same = fractures[:-1] == fractures[1:]
        f, u, v = fractures[:-1][same], ids[:-1][same], ids[1:][same]
        length = np.diff(positions)[same] * self.clipped_lengths[f]
        c = (
            self.fluid_density
            * self.gravity
            * self.apertures[f] ** 3
            / (12 * self.fluid_viscosity * length)
        )
        self._pipes = (u, v, c)
        start, end = self.endpoints[f, 0], self.endpoints[f, 1]
        self._pipe_segments = np.stack(  # Shape (pipes, 2, 2): pipe, start/end, x/y
            [
                start + positions[:-1][same, None] * (end - start),
                start + positions[1:][same, None] * (end - start),
            ],
            axis=1,
        )

        # Mass balance at node i: sum over pipes (i, j) of c_ij (h_i - h_j) = 0, with boundary heads moved to the right-hand side
        rows, cols, vals = [], [], []
        b = np.zeros(m)
        for a, z in ((u, v), (v, u)):
            inner = a >= 0
            rows += [a[inner]]
            cols += [a[inner]]
            vals += [c[inner]]
            both = inner & (z >= 0)
            rows += [a[both]]
            cols += [z[both]]
            vals += [-c[both]]
            to_boundary = inner & (z < 0)
            np.add.at(
                b,
                a[to_boundary],
                c[to_boundary]
                * np.where(z[to_boundary] == LEFT, head_left, head_right),
            )

        self.A = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(m, m),
        )
        self.b = b
        self.heads = self.total_flow = None  # Any previous solution is now stale

    def solve(self) -> float:
        """Solves Ah = b for the hydraulic head at every node, storing it in `heads`, and returns the total flow rate (per unit depth)
        from the left boundary to the right boundary. Calls analyze() with its default boundary heads if it has not yet been run."""
        if self.A is None:
            self.analyze()
        self.heads = (
            np.atleast_1d(spsolve(self.A.tocsc(), self.b))
            if len(self.b)
            else np.zeros(0)
        )

        # Indexing with LEFT = -1 and RIGHT = -2 picks up the boundary heads appended at the end
        h = np.concatenate([self.heads, [self.head_right, self.head_left]])
        u, v, c = self._pipes
        q = c * (h[u] - h[v])  # Flow from u to v along each pipe
        self.total_flow = q[u == LEFT].sum() - q[v == LEFT].sum()
        return self.total_flow

    def plot(
        self, ax: plt.Axes | None = None, show_nodes: bool | None = None
    ) -> plt.Axes:
        """Draws a representation of the fracture network.

        Fractures pruned away as isolated are drawn faint gray. If solve() has been run, the pipes carrying flow are coloured by
        hydraulic head, leaving dead-end fracture tips in mid gray; otherwise all retained fractures are drawn in blue. Nodes are
        marked when show_nodes is True, or by default when there are at most 1000 of them. Returns the axes drawn on.
        """
        from matplotlib.collections import LineCollection
        from matplotlib.colors import LinearSegmentedColormap

        if ax is None:
            _, ax = plt.subplots(
                figsize=(8, 8 * self.height / self.width), layout="constrained"
            )
        if self.clusters is None:
            self.cluster_and_prune()
        solved = self.heads is not None

        ax.add_collection(
            LineCollection(
                self.endpoints[~self.active], colors="#c3c2b7", linewidths=0.6
            )
        )
        ax.add_collection(
            LineCollection(
                self.endpoints[self.active],
                colors="#8a8984" if solved else "#2a78d6",
                linewidths=1.0,
            )
        )

        if solved:
            # Colour each pipe by the mean head of its two ends; the light end of the ramp is dropped so it stays visible on white
            h = np.concatenate([self.heads, [self.head_right, self.head_left]])
            u, v, _ = self._pipes
            cmap = LinearSegmentedColormap.from_list(
                "head", plt.cm.Blues(np.linspace(0.3, 1.0, 256))
            )
            pipes = LineCollection(
                self._pipe_segments, array=(h[u] + h[v]) / 2, cmap=cmap, linewidths=1.4
            )
            pipes.set_clim(
                min(self.head_left, self.head_right),
                max(self.head_left, self.head_right),
            )
            ax.add_collection(pipes)
            ax.figure.colorbar(pipes, ax=ax, label="Hydraulic head", shrink=0.8)
            ax.set_title(
                f"{len(self.lengths)} fractures, {len(self.nodes)} nodes; total flow {self.total_flow:.3e}",
                loc="left",
            )
        else:
            ax.set_title(
                f"{len(self.lengths)} fractures, {len(self.clusters)} connected clusters",
                loc="left",
            )

        if show_nodes or (show_nodes is None and len(self.nodes) <= 1000):
            ax.scatter(*self.nodes.T, s=8, color="#0b0b0b", zorder=3, linewidths=0)

        # Domain outline, with the fixed-head boundaries drawn heavier than the impermeable top and bottom
        ax.plot(
            [0, self.width, self.width, 0, 0],
            [0, 0, self.height, self.height, 0],
            color="#52514e",
            linewidth=0.8,
        )
        ax.vlines([0, self.width], 0, self.height, colors="#0b0b0b", linewidths=2.5)

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return ax
