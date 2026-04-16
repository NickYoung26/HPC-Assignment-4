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
Ising model (discrete spin flips) and the XY model (continuous angle
updates). One Monte Carlo sweep consists of N = L*L single-site
update attempts.

Date: 15/04/2026

Author: Nicholas Young
"""
import numpy as np

from isingmod import IsingModel
from xymod import XYModel
from analysis import vortex_density

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

def sweep_xy(model: XYModel, beta: float, max_angle_step: float = np.pi) -> int:
    """
    Perform one full Metropolis sweep over all XY lattice sites.

    Each site is visited once. A random angle perturbation drawn uniformly
    from [-max_angle_step, +max_angle_step] is proposed and accepted with
    probability min(1, exp(-beta * dE)).

    Args:
        model: XYModel instance to update in place.
        beta: Inverse temperature 1 / (k_B * T).
        max_angle_step: Maximum magnitude of proposed angle change (radians).

    Returns:
        Number of accepted angle updates during the sweep.
    """
    size = model.size
    rng = model._rng
    accepted = 0

    # Visit sites in random order
    rows = rng.integers(0, size, size=model.n_sites)
    cols = rng.integers(0, size, size=model.n_sites)
    angle_steps = rng.uniform(-max_angle_step, max_angle_step, size=model.n_sites)
    log_randoms = np.log(rng.uniform(0.0, 1.0, size=model.n_sites))

    # Spin flipper and accepted counter
    for k in range(model.n_sites):
        row, col = rows[k], cols[k]
        new_angle = (model.angles[row, col] + angle_steps[k]) % (2.0 * np.pi)
        delta_e = model.delta_energy(row, col, new_angle)
        if delta_e <= 0.0 or log_randoms[k] < -beta * delta_e:
            model.angles[row, col] = new_angle
            accepted += 1

    return accepted

def equilibrate(model, beta: float, n_equilibration: int, model_type: str = "ising"):
    """
    Run equilibration sweeps to thermalise the model.
    Same process for XY and Ising

    Args:
        model: IsingModel or XYModel
        beta: Inverse temperature.
        n_equilibration: Number of sweeps to discard.
        model_type: Either 'ising' or 'xy'.
    """
    sweep_fn = sweep_ising if model_type == "ising" else sweep_xy
    for _ in range(n_equilibration):
        sweep_fn(model, beta)

def collect_samples(
    model,
    beta: float,
    n_samples: int,
    sample_interval: int = 1,
    model_type: str = "ising",
) -> dict:
    """
    Collect thermodynamic observables after equilibration.
    Same process for XY and Ising

    Args:
        model: IsingModel or XYModel instance (already equilibrated).
        beta: Inverse temperature.
        n_samples: Number of samples to collect.
        sample_interval: Sweeps between successive samples.
        model_type: Either 'ising' or 'xy'.

    Returns:
        Dictionary with arrays:
            'energy': energy per site for each sample.
            'energy_sq': (energy per site)^2 for each sample.
            For Ising also 'magnetisation'.
            For XY also 'correlations': dict mapping r_frac -> array.
    """
    sweep_fn = sweep_ising if model_type == "ising" else sweep_xy

    energies = np.zeros(n_samples)
    energies_sq = np.zeros(n_samples)

    if model_type == "ising":
        magnetisations = np.zeros(n_samples)
    else:
        r_fracs = np.linspace(1.0 / model.size, 0.5, num=10)
        correlations = {r: np.zeros(n_samples) for r in r_fracs}

    if model_type == "xy":
        from analysis import vortex_density
        temperature = 1.0 / beta
        helicity = np.zeros(n_samples)
        vortices = np.zeros(n_samples)

    for i in range(n_samples):
        for _ in range(sample_interval):
            sweep_fn(model, beta)
        e = model.energy_per_site()
        energies[i] = e
        energies_sq[i] = e * e

        if model_type == "ising":
            magnetisations[i] = model.magnetisation()
        else:
            for r in r_fracs:
                correlations[r][i] = model.spin_correlation(r)
            vortices[i] = vortex_density(model.angles)

    result = {"energy": energies, "energy_sq": energies_sq}
    if model_type == "ising":
        result["magnetisation"] = magnetisations
    else:
        result["correlations"] = correlations
        result["vortex_density"] = vortices
    return result
