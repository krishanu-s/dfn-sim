"""Simulator for Discrete Fracture Networks."""

import numpy as np
from pip._vendor.certifi import where


class DFNModel:
    """A two-dimensional discrete fracture network is a set of fractures within a bounded (rectangular) two-dimensional domain.
    Each fracture is a line segment defined by its length, midpoint location, orientation, and aperture. Field studies indicate that
    these values should be generated according to the following distributions:
        - Midpoints are chosen so that a global density criterion is met. A starting example is a Poisson point process, i.e. so that
        the number of fractures within a box of area V follows such a process with mean proportional to V.
        - Length L follows a power law distribution with probability density function ~ L^{-α-1}
        where 1 <= α <= 3.
        - Orientation θ clusters around a mean orientation μ, with probability density function ~ exp(k * cos(θ - μ))/(2πI_0(k))
        where I_0 is the modified Bessel function of zero order and k is a constant.
        - Aperture A can follow a log-normal distribution, or be power-law dependent on the length L.

    Analysis is aimed at calculating fluid flow from the left boundary of the box to the right boundary, assuming a given pressure differential
    between the two sides. This can be deduced from the higher-resolution question of solving a linear equation of the form Ah = b, where A
    encodes the cluster behavior of the network, b is defined by connections to the boundaries, and h is a vector of length equal to the number
    of nodes and gives the hydraulic head at all nodes.
    """

    def __init__(
        self,
        width: float,  # Width of the bounding box
        height: float,  # Height of the bounding box
        # TODO Add parameters for the Poisson point process defining midpoint locations
        alpha: float,  # Power law exponent governing the distribution of fracture lengths
        mu: float,  # Mean of orientation distribution function
        k: float,  # Variance of orientation distribution function
        # TODO Add parameters for the aperture width distribution
    ):
        # TODO set all of the internal parameters
        self._generate_dfn()
        raise NotImplementedError

    def _generate_dfn(self):
        """Stochastically generates a DFN instance, storing the results within the class."""
        raise NotImplementedError

    def cluster_and_prune(self):
        """Clusters overlapping fractures, and prunes away any clusters which are isolated (i.e. which do not connect to any boundaries).
        Finally, defines a set of `nodes' which are intersection points between fractures. Stores the named clusters and nodes within the class.
        """
        raise NotImplementedError

    def analyze(self):
        """Analyzes the discrete fracture network instance, generating the matrix A and the vector b as described in the class definition
        (and storing them within the class)."""
        self.cluster_and_prune()
        raise NotImplementedError
