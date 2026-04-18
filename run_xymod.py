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

MPI-parallel Metropolis Monte Carlo simulation of the 2D XY model.

Each MPI rank acts as an independent walker, sampling the XY model at
each temperature in the range [T_MIN, T_MAX]. Results (energy per site,
specific heat, spin correlations) are gathered to rank 0, which writes
them to a NumPy .npz archive.

Date: 15/04/2026

Author: Nicholas Young
"""

import argparse
import time

import numpy as np # pylint: disable=import-error
from mpi4py import MPI # pylint: disable=import-error

from xymod import XYModel
from metropolis_rw import equilibrate, collect_samples
from analysis import specific_heat, mean_energy, mean_correlation, combined_walker_results

# Temperature range for the XY model (in units of k_B / J)
T_MIN = 0.5
T_MAX = 1.5

# Fractional separations at which to evaluate spin correlations
N_CORR_POINTS = 10


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="2D XY model MPI simulation")
    parser.add_argument("--size", type=int, default=32, help="Lattice size L")
    parser.add_argument("--n-temps", type=int, default=20, help="Number of temperature points")
    parser.add_argument("--n-equil", type=int, default=2000, help="Equilibration sweeps")
    parser.add_argument("--n-samples", type=int, default=2500, help="Production samples per walker")
    parser.add_argument("--sample-interval", type=int, default=20, help="Sweeps between samples")
    parser.add_argument("--outfile", type=str, default=r"xy_results.npz", help="Output filename")
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
    Run equilibration and production for one temperature on this rank.

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
    seed = abs(hash((rank, temperature, "xy"))) % (2 ** 31)
    rng = np.random.default_rng(seed)
    model = XYModel(size=size, coupling=1.0, rng=rng)
    beta = 1.0 / temperature

    equilibrate(model, beta, n_equil, model_type="xy")
    return collect_samples(model, beta, n_samples, sample_interval, model_type="xy")

def main():
    """Main entry point: distribute work across MPI ranks and gather results."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    args = parse_args()
    temperatures = np.linspace(T_MIN, T_MAX, args.n_temps)
    r_fracs = np.linspace(1.0 / args.size, 0.5, num=N_CORR_POINTS)

    if rank == 0:
        print(f"XY model: L={args.size}, {n_ranks} MPI ranks, "
              f"{args.n_temps} temperatures, {args.n_samples} samples/rank")
        t_start = time.time()

    all_energies = np.zeros(args.n_temps)
    all_cv = np.zeros(args.n_temps)

    # spin correlations[t_idx, r_idx]
    all_correlations = np.zeros((args.n_temps, len(r_fracs)))
    all_vortex = np.zeros(args.n_temps)

    for t_idx, temperature in enumerate(temperatures):
        beta = 1.0 / temperature

        local_result = simulate_temperature(
            args.size, temperature, args.n_equil,
            args.n_samples, args.sample_interval, rank
        )

        all_results = comm.gather(local_result, root=0)

        if rank == 0:
            combined = combined_walker_results(all_results)
            n_sites = args.size ** 2
            all_energies[t_idx] = mean_energy(combined["energy"])
            all_cv[t_idx] = specific_heat(combined["energy"], beta, n_sites)
            all_vortex[t_idx] = float(np.mean(combined["vortex_density"]))

            for r_idx, r in enumerate(r_fracs):
                all_correlations[t_idx, r_idx] = mean_correlation(
                    combined["correlations"][r]
                )

            print(f"  T={temperature:.3f}  <e>={all_energies[t_idx]:.4f}"
                  f"  Cv={all_cv[t_idx]:.4f} cor={all_correlations[t_idx, r_idx]:.3f}" # pylint: disable=undefined-loop-variable
                  f"  n_v={all_vortex[t_idx]:.4f}")

    if rank == 0:
        elapsed = time.time() - t_start
        print(f"\nSimulation complete in {elapsed:.1f}s")
        np.savez(
            args.outfile,
            temperatures=temperatures,
            energies=all_energies,
            specific_heat=all_cv,
            correlations=all_correlations,
            r_fracs=r_fracs,
            vortex_density=all_vortex,
            size=np.array([args.size]),
            n_ranks=np.array([n_ranks]),
            elapsed=np.array([elapsed]),
        )
        print(f"Results saved to {args.outfile}")

if __name__ == "__main__":
    main()
