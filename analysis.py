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

Thermodynamic observable calculations from Monte Carlo metropolis samples.

Provides functions to gather and then compute the specific heat capacity, 
magnetisation and mean energy.


Date: 15/04/2026

Author: Nicholas Young

"""

import numpy as np

def combined_walker_results(results: list) -> dict:
    """
    Combine results from multiple independent walkers.

    Concatenates observable arrays from each walker along axis 0
    so that derived quantities (means, variances) use the full
    combined sample.

    Args:
        results: List of result dictionaries, one per walker, as
            returned by metropolis.collect_samples.

    Returns:
        Single result dictionary with concatenated arrays.
    """
    combined = {}
    keys = results[0].keys()
    for key in keys:
        if key == "correlations":
            r_fracs = results[0]["correlations"].keys()
            combined["correlations"] = {
                r: np.concatenate([res["correlations"][r] for res in results])
                for r in r_fracs
            }
        else:
            combined[key] = np.concatenate([res[key] for res in results])
    return combined

def specific_heat(energies: np.ndarray, beta: float, n_sites: int) -> float:
    """
    Compute the specific heat per site from energy samples.

    Uses the fluctuation–dissipation relation:
        C_v = beta^2 * N * (< e^2 > - < e >^2)

    where e = E / N is the energy per site.

    Args:
        energies: 1-D array of sampled energies per site.
        beta: Inverse temperature 1 / (k_B * T).
        n_sites: Total number of lattice sites N = L * L.

    Returns:
        Specific heat per site C_v / N.
    """
    mean_e = np.mean(energies)
    mean_e_sq = np.mean(energies ** 2)
    variance = mean_e_sq - mean_e ** 2
    return float(beta ** 2 * n_sites * variance)


def mean_energy(energies: np.ndarray) -> float:
    """
    Compute the mean energy per site.

    Args:
        energies: 1-D array of sampled energies per site.

    Returns:
        Mean energy per site <e>.
    """
    return float(np.mean(energies))


def mean_magnetisation(magnetisations: np.ndarray) -> float:
    """
    Compute the mean absolute magnetisation per site.

    Args:
        magnetisations: 1-D array of sampled |m| values.

    Returns:
        Mean absolute magnetisation <|m|>.
    """
    return float(np.mean(np.abs(magnetisations)))
