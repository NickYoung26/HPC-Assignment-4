#!/opt/software/anaconda/python-3.10.9/bin/python

# =============================================================================
# MIT License
#
# Copyright (c) 2026 Nicholas Young
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

"""
Version: Python 3.10.9

2D nearest-neighbour Ising model on an L x L square lattice.

Hamiltonian is:
    H = -J * sum_{<i,j>} s_i * s_j

where s_i in {-1, +1} and the sum runs over nearest-neighbour pairs.
Periodic boundary conditions are applied, where monte-carlo randdom
walkers are implemented to evaluate lattice points.

Date: 15/04/2026

Author: Nicholas Young
"""

import numpy as np

"""
Ideas (delete later):

Create Ising model as a class of functions which describe the core principles
of the model.

To achieve that we will need to realise the following functions in code:

Lattice creation and random spin application.

Hamiltonian calculations (local, spin flip affects)

magnetisation calculation.

Energies are bound through spin (+1,-1) coupling between lattice sites
where two of the same spin are energetically weaker than two of
opposing spins. This is from Pauli exclusion.

For ferromagnetism J = 1 (J is coupling strength)

magnetisation and E are needed to find the critical temp from what I can tell.

Thats where ferromagnetism breaks down causing paramagnetism.

From reading up these energy values need to be evaluated
each time metropolis tries to make a spin flip.
"""

class IsingModel:

    """
    2D Ising model on a square lattice.

    Attributes:
        size: Linear dimension L of the L x L lattice.
        coupling: Exchange coupling constant J.
        n_sites: Total number of lattice sites (L * L).
        spins: Array of spin values, each in {-1, +1}.
    """

    def __init__(self, size: int, coupling: float = 1.0, rng: np.random.Generator = None):
        """
        Initialise the Ising model with lattice construction 
        and random spin configuration.

        Args:
            size: Linear dimension L of the square lattice.
            coupling: Exchange coupling constant J (default 1.0).
            rng: NumPy random generator for reproducibility.
        """
        self.size = size # set lattice size
        self.coupling = coupling
        self.n_sites = size * size
        self._rng = rng if rng is not None else np.random.default_rng()
        self.spins = self._rng.choice([-1, 1], size=(size, size)).astype(np.int8)

    # Energy Terms

    def total_energy(self) -> float:
        """
        Compute the total energy of the current spin configuration.

        Uses vectorised roll operations to sum over all nearest-neighbour
        pairs without explicit loops.

        Returns:
            Total energy E of the configuration.
        """
        spins = self.spins
        # Sum contributions from right,left and up,down neighbours.
        # Using np.roll as recomended to get neighbour vals.

        neighbour_sum = (
            np.roll(spins, -1, axis=1)  # right
            + np.roll(spins, 1, axis=1)  # left
            + np.roll(spins, -1, axis=0)  # down
            + np.roll(spins, 1, axis=0)  # up
        )
        return -0.5 * self.coupling * float(np.sum(spins * neighbour_sum)) 
        # (0.5 because each bond counted twice.)

    # Other terms

    def magnetisation(self) -> float:
        """
        Compute the mean magnetisation per site.

        Returns:
            Absolute magnetisation per site.
        """
        return float(np.abs(np.mean(self.spins)))









