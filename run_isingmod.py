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

Used to run Ising model in parrallel.

Date: 15/04/2026

Author: Nicholas Young
"""



"""
Ideas :

Can import ising and metropolis to compute them

probs best to use a parser so I can easily change 
the important variables associated with metropolis
and ising model.

Ising variables:
lattice size
temp range 
temp iterations
anything else?

metropolis variables:
no. of walkers?
need to see whqat else is needed.
"""
import argparse
import time

import numpy as np
from mpi4py import MPI

from isingmod import IsingModel
from metropolis_rw import equilibrate, sweep_ising

# Temperature range for the Ising model (in units of k_B / J)
T_MIN = 1.0
T_MAX = 3.0 

def parse_args() -> argparse.Namespace:
    """
    Parser command-line arguments for solving metropolis-ising.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="2D Ising model MPI simulation")
    parser.add_argument("--size", type=int, default=32, help="Lattice size L")
    parser.add_argument("--n-temps", type=int, default=20, help="Number of temperature points")
    parser.add_argument("--n-equil", type=int, default=2000, help="Equilibration sweeps")
    parser.add_argument("--n-samples", type=int, default=2500, help="Production sweeps per walker")
    parser.add_argument("--sample-interval", type=int, default=5, help="Sweeps between samples")
    parser.add_argument("--outfile", type=str, default=r"ising_results.npz", help="Output filename")
    return parser.parse_args()

def simulate_temperature(
    size: int,
    temperature: float,
    n_equil: int,
    n_samples: int,
    sample_interval: int,
    rank: int,
) -> dict:
    """
    Intialise isingmod.py and metropolis variables to run 
    equilibration and production for a given temperature
    on this rank.

    Args:
        size: Lattice size L.
        temperature: Temperature T in units of k_B / J.
        n_equil: Number of equilibration sweeps.
        n_samples: Number of production samples to collect.
        sample_interval: Sweeps between successive samples.
        rank: MPI rank (used to seed the RNG uniquely).

    Returns:
        Dictionary of sampled observables.
    """
    seed = abs(hash((rank, temperature))) % (2 ** 31)
    rng = np.random.default_rng(seed)
    model = IsingModel(size=size, coupling=1.0, rng=rng) # J = 1 
    beta = 1.0 / temperature

    equilibrate(model, beta, n_equil, model_type="ising")
    return collect_samples(model, beta, n_samples, sample_interval, model_type="ising")


