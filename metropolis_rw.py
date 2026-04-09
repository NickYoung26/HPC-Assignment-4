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

Metropolis Monte Carlo sampler for 2D spin lattice models.

Implements the standard Metropolis algorithm for both the
Ising model (discrete spin flips). One Monte Carlo sweep consists of N = L*L single-site
update attempts.

Date: 15/04/2026

Author: Nicholas Young


we could use this method for XY model as well?
"""
import numpy as np

from isingmod import IsingModel

def sweep_ising(model: IsingModel, beta: float) -> int:
    """
    Perform one full Metropolis sweep over all Ising lattice sites.

    Each site is visited once in a random order. A spin flip is accepted
    with probability min(1, exp(-beta * dE)).

    Args:
        model: IsingModel instance to update in place.
        beta: Inverse temperature 1 / (k_B * T).

    Returns:
        Number of accepted spin flips during the sweep.
    """
    size = model.size
    rng = model._rng
    accepted = 0

    # Visit sites in random order (systematic bias)
    rows = rng.integers(0, size, size=model.n_sites)
    cols = rng.integers(0, size, size=model.n_sites)
    log_randoms = np.log(rng.uniform(0.0, 1.0, size=model.n_sites))

    # Spin flipper and accepted counter
    for k in range(model.n_sites):
        row, col = rows[k], cols[k]
        delta_e = model.delta_energy(row, col)
        if delta_e <= 0.0 or log_randoms[k] < -beta * delta_e:
            model.spins[row, col] *= -1
            accepted += 1

    return accepted


