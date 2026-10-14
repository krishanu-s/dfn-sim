import code

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import cg, eigs, spsolve

from dfn import DFNModel


def dfn_example():
    """Instantiates a discrete fracture network, and solves for the hydraulic head at
    all nodes. Generates a visualization of the"""
    # Instantiate a model
    model = DFNModel(
        width=3.0, height=3.0, density=60.0, alpha=2.5, mu=0, k=0, l_min=0.2
    )

    # Analyze to produce flow matrix A and and solve
    model.analyze()
    total_flow = model.solve()

    # Make a plot
    # ax = model.plot()

    # Generate eigenvalues to check the condition number
    (n,) = model.b.shape
    eigenvals, eigenvecs = eigs(model.A, n - 2)
    eigenvals.sort()
    eigenvals = eigenvals[::-1]

    # Make a figure with the fracture network visualized at left, and the
    # distribution of eigenvalues at right
    fig, (ax1, ax2) = plt.subplots(1, 2)
    model.plot(ax1)
    ax2.scatter(range(0, n - 2), eigenvals, s=2)
    ax2.set_yscale("log")
    ax2.set_title("Eigenvalue distribution")

    # Drop an interactive shell here
    variables = globals().copy()
    variables.update(locals())
    shell = code.InteractiveConsole(variables)
    shell.interact()

    plt.show()


if __name__ == "__main__":
    dfn_example()
