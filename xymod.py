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

2D nearest-neighbour XY model on an L x L square lattice.

The Hamiltonian is:
    H = -J * sum_{<i,j>} cos(theta_i - theta_j)

where theta_i in [0, 2*pi) is the in-plane spin angle at site i.
Periodic boundary conditions are applied in both dimensions.

Date: 15/04/2026

Author: Nicholas Young
"""

import numpy as np


class XYModel:

    """
    Classical 2D XY model on a square lattice.

    Attributes:
        size: Linear dimension L of the L x L lattice.
        coupling: Exchange coupling constant J.
        n_sites: Total number of lattice sites (L * L).
        angles: Array of spin angles in [0, 2*pi).
    """

    def __init__(self, size: int, coupling: float = 1.0, rng: np.random.Generator = None):
        """
        Initialise the XY model with random spin angles.

        Args:
            size: Linear dimension L of the square lattice.
            coupling: Exchange coupling constant J (default 1.0).
            rng: NumPy random generator for reproducibility.
        """
        self.size = size
        self.coupling = coupling
        self.n_sites = size * size
        self._rng = rng if rng is not None else np.random.default_rng()
        self.angles = self._rng.uniform(0.0, 2.0 * np.pi, size=(size, size))

    def total_energy(self) -> float:
        """
        Compute the total energy of the current angle configuration.

        Uses vectorised roll operations over nearest-neighbour pairs.

        Returns:
            Total energy E of the configuration.
        """
        angles = self.angles
        energy = (
            np.cos(angles - np.roll(angles, -1, axis=1))  # right
            + np.cos(angles - np.roll(angles, 1, axis=1))  # left
            + np.cos(angles - np.roll(angles, -1, axis=0))  # down
            + np.cos(angles - np.roll(angles, 1, axis=0))  # up
        )
        return -0.5 * self.coupling * float(np.sum(energy))

    def site_energy(self, row: int, col: int) -> float:
        """
        Compute the local energy contribution of a single spin.

        Args:
            row: Row index of the site.
            col: Column index of the site.

        Returns:
            Local energy contribution at (row, col).
        """
        size = self.size
        theta = self.angles[row, col]
        neighbour_angles = np.array([
            self.angles[(row + 1) % size, col],
            self.angles[(row - 1) % size, col],
            self.angles[row, (col + 1) % size],
            self.angles[row, (col - 1) % size],
        ])
        return -self.coupling * float(np.sum(np.cos(theta - neighbour_angles)))

    def delta_energy(self, row: int, col: int, new_angle: float) -> float:
        """
        Energy change from updating angle at (row, col) to new_angle.

        Args:
            row: Row index of the candidate spin.
            col: Column index of the candidate spin.
            new_angle: Proposed new angle in [0, 2*pi).

        Returns:
            Change in total energy dE for the proposed update.
        """
        size = self.size
        neighbour_angles = np.array([
            self.angles[(row + 1) % size, col],
            self.angles[(row - 1) % size, col],
            self.angles[row, (col + 1) % size],
            self.angles[row, (col - 1) % size],
        ])
        old_e = -self.coupling * float(np.sum(np.cos(self.angles[row, col] - neighbour_angles)))
        new_e = -self.coupling * float(np.sum(np.cos(new_angle - neighbour_angles)))
        return new_e - old_e




