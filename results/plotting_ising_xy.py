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
Plot thermodynamic data from Ising and XY model simulations.

Reads .npz files and generates relevant figures from contained data.
"""

import argparse
import os

import numpy as np # pylint: disable=import-error
import matplotlib.pyplot as plt # pylint: disable=import-error
import matplotlib.ticker as ticker # pylint: disable=import-error consider-using-from-import
from scipy.optimize import curve_fit # pylint: disable=import-error

# Ising critical temperature (exact Onsager result)
TC_ISING = 2.0 / np.log(1.0 + np.sqrt(2.0))  # ≈ 2.269 J/k_B

FIGURE_DPI = 150
FIGURE_DIR = "figures"

def label_from_file(filepath: str) -> str:
    """
    Extract a short legend label from a results filename.

    Args:
        filepath: Path to a .npz results file.

    Returns:
        Label string, e.g. 'L=32'.
    """
    data = np.load(filepath)
    size = int(data["size"][0])
    return f"L = {size}"

def plot_ising(filepaths: list):
    """
    Generate Ising model figures: energy per site and specific heat.

    Args:
        filepaths: List of .npz result file paths for different lattice sizes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_e, ax_cv = axes

    for fpath in filepaths:
        data = np.load(fpath)
        label = label_from_file(fpath)
        temps = data["temperatures"]
        energies = data["energies"]
        cv = data["specific_heat"]

        ax_e.plot(temps, energies, marker="o", markersize=3, label=label)
        ax_cv.plot(temps, cv, marker="o", markersize=3, label=label)

    for ax in axes:
        ax.axvline(TC_ISING, color="gray", linestyle="--", linewidth=0.9, label=r"$T_c$")
        ax.set_xlabel(r"T = $k_B T \, / \, J$", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid()
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    ax_e.set_ylabel(r"$\langle e \rangle \, / \, J$", fontsize=12)
    ax_e.set_title("Ising Model: Mean Energy", fontsize=11)

    ax_cv.set_ylabel(r"$C_v \, / \, k_B$", fontsize=12)
    ax_cv.set_title("Ising Model: Specific Heat", fontsize=11)

    fig.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    outpath = os.path.join(FIGURE_DIR, "ising_observables.png")
    fig.savefig(outpath, dpi=FIGURE_DPI)
    print(f"Saved {outpath}")
    plt.close(fig)

def plot_ising_magnetisation(filepaths: list):
    """
    Plot Ising magnetisation per site vs temperature.

    Args:
        filepaths: List of .npz result file paths.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for fpath in filepaths:
        data = np.load(fpath)
        if "magnetisation" not in data:
            continue
        label = label_from_file(fpath)
        ax.plot(data["temperatures"], data["magnetisation"],
                marker="o", markersize=3, label=label)

    ax.axvline(TC_ISING, color="gray", linestyle="--", linewidth=0.9, label=r"$T_c$")
    ax.set_xlabel(r"T = $k_B T \, / \, J$", fontsize=12)
    ax.set_ylabel(r"$\langle |m| \rangle$", fontsize=12)
    ax.set_title("Ising Model: Mean Magnetisation", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid()
    fig.tight_layout()
    outpath = os.path.join(FIGURE_DIR, "ising_magnetisation.png")
    fig.savefig(outpath, dpi=FIGURE_DPI)
    print(f"Saved {outpath}")
    plt.close(fig)

def plot_xy(filepaths: list):
    """
    Generate XY model figures: specific heat vs temperature.

    Args:
        filepaths: List of .npz result file paths.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for fpath in filepaths:
        data = np.load(fpath)
        label = label_from_file(fpath)
        ax.plot(data["temperatures"], data["specific_heat"],
                marker="o", markersize=3, label=label)
    ax.set_xlabel(r"T = $k_B T \, / \, J$", fontsize=12)
    ax.set_ylabel(r"$C_v \, / \, k_B$", fontsize=12)
    ax.set_title("XY Model: Specific Heat", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid()
    fig.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    outpath = os.path.join(FIGURE_DIR, "xy_specific_heat.png")
    fig.savefig(outpath, dpi=FIGURE_DPI)
    print(f"Saved {outpath}")
    plt.close(fig)

def plot_xy_correlations(filepaths: list): # pylint: disable=too-many-locals
    """
    Plot XY spin–spin correlation vs fractional distance at several temperatures.

    Args:
        filepaths: List of .npz result file paths.
    """
    for fpath in filepaths:
        data = np.load(fpath)
        label = label_from_file(fpath)
        size = int(data["size"][0])
        temps = data["temperatures"]
        r_fracs = data["r_fracs"]
        correlations = data["correlations"]  # shape (n_temps, n_r)

        fig, ax = plt.subplots(figsize=(7, 4.5))

        # Select a handful of temperatures to display
        n_temps = len(temps)
        indices = np.linspace(0, n_temps - 1, min(7, n_temps), dtype=int)
        cmap = plt.get_cmap("rainbow")
        colors = [cmap(i / max(len(indices) - 1, 1)) for i in range(len(indices))]

        for plot_idx, t_idx in enumerate(indices):
            ax.plot(
                r_fracs,
                correlations[t_idx],
                marker="o",
                markersize=3,
                color=colors[plot_idx],
                label=f"T = {temps[t_idx]:.2f}",
            )

        ax.set_xlabel(r"$r \, / \, L$", fontsize=12)
        ax.set_ylabel(r"$C(r) = \langle \cos(\theta_i - \theta_j) \rangle$", fontsize=12)
        ax.set_title(f"XY Model Spin Correlation ({label})", fontsize=11)
        ax.legend(fontsize=8, ncol=2)
        ax.grid()
        fig.tight_layout()
        outpath = os.path.join(FIGURE_DIR, f"xy_correlation_L{size}.png")
        fig.savefig(outpath, dpi=FIGURE_DPI)
        print(f"Saved {outpath}")
        plt.close(fig)

def plot_xy_vortex_density(filepaths):
    """
    Plot vortex density vs temperature for the XY model.

    Below T_BKT vortices are tightly bound in pairs (low density).
    Above T_BKT they unbind and proliferate (rising density).

    Args:
        filepaths: List of .npz result file paths.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for fpath in filepaths:
        data = np.load(fpath)
        if "vortex_density" not in data:
            continue
        ax.plot(data["temperatures"], data["vortex_density"],
                marker="o", markersize=3, label=label_from_file(fpath))
    ax.set_xlabel(r"T = $k_B T \, / \, J$", fontsize=12)
    ax.set_ylabel(r"Vortex Density $n_v$", fontsize=12)
    ax.set_title("XY model: Topological Vortex Density", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid()
    fig.tight_layout()
    outpath = os.path.join(FIGURE_DIR, "xy_vortex_density.png")
    fig.savefig(outpath, dpi=FIGURE_DPI)
    print("Saved " + outpath)
    plt.close(fig)

def _power_law(r, eta, amplitude):
    """
    Power-law model C(r) = amplitude * r^(-eta).

    Args:
        r: Separations in lattice units.
        eta: Decay exponent.
        amplitude: Prefactor.

    Returns:
        Array of model values.
    """
    return amplitude * r ** (-eta)

def _exponential(r, xi, amplitude):
    """
    Exponential model C(r) = amplitude * exp(-r / xi).

    Args:
        r: Separations in lattice units.
        xi: Correlation length.
        amplitude: Prefactor.

    Returns:
        Array of model values.
    """
    return amplitude * np.exp(-r / xi)

def _fit_correlations(r_fracs, correlations, size):
    """
    Fit power-law and exponential models to a single correlation curve.

    Args:
        r_fracs: 1-D array of fractional separations r/L.
        correlations: 1-D array of mean C(r) values.
        size: Lattice size L.

    Returns:
        Tuple (eta, xi, preferred) where eta is the power-law exponent,
        xi is the exponential correlation length, and preferred is either
        'power_law' or 'exponential'. Failed fits return NaN.
    """
    r = r_fracs * size

    try:
        p_opt, _ = curve_fit(_power_law, r, correlations,
                             p0=[0.25, 1.0], maxfev=2000,
                             bounds=([0, 0], [2, 10]))
        pl_resid = float(np.sqrt(np.mean((_power_law(r, *p_opt) - correlations) ** 2)))
        eta = float(p_opt[0])
    except RuntimeError:
        pl_resid = np.inf
        eta = np.nan

    try:
        e_opt, _ = curve_fit(_exponential, r, correlations,
                             p0=[size / 4, 1.0], maxfev=2000,
                             bounds=([0.1, 0], [size * 10, 10]))
        exp_resid = float(np.sqrt(np.mean((_exponential(r, *e_opt) - correlations) ** 2)))
        xi = float(e_opt[0])
    except RuntimeError:
        exp_resid = np.inf
        xi = np.nan

    preferred = "power_law" if pl_resid <= exp_resid else "exponential"
    return eta, xi, preferred

def _compute_fit_arrays(filepaths):
    """
    Compute eta and xi arrays for all lattice sizes.

    Shared helper used by both correlation plot functions to avoid
    duplicating the fitting loop.

    Args:
        filepaths: List of .npz result file paths.

    Returns:
        List of dicts with keys 'label', 'temps', 'eta_arr', 'xi_arr'.
    """
    results = []
    for fpath in filepaths:
        data = np.load(fpath)
        if "correlations" not in data or "r_fracs" not in data:
            continue
        size = int(data["size"][0])
        temps = data["temperatures"]
        r_fracs = data["r_fracs"]
        correlations = data["correlations"]

        eta_arr = np.zeros(len(temps))
        xi_arr = np.zeros(len(temps))
        for t_idx in range(len(temps)):
            eta, xi, _ = _fit_correlations(r_fracs, correlations[t_idx], size)
            eta_arr[t_idx] = eta if not np.isnan(eta) else np.nan
            xi_arr[t_idx] = xi if not np.isnan(xi) else np.nan

        results.append({
            "label": label_from_file(fpath),
            "temps": temps,
            "eta_arr": eta_arr,
            "xi_arr": xi_arr,
        })
    return results

def plot_xy_correlation_exponent(filepaths):
    """
    Plot the power-law decay exponent eta vs temperature.

    Fits C(r) ~ r^(-eta) at each temperature and plots the exponent eta.
    Below T_BKT eta is small and slowly increasing. At T_BKT it reaches
    the universal value eta = 0.25 predicted by BKT theory.

    Args:
        filepaths: List of .npz result file paths.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for res in _compute_fit_arrays(filepaths):
        ax.plot(res["temps"], res["eta_arr"],
                marker="o", markersize=3, label=res["label"])

    ax.axhline(0.25, color="red", linestyle=":", linewidth=0.9,
               label=r"$\eta$ = 0.25 (universal BKT value)")
    ax.set_xlabel(r"T = $k_B T \, / \, J$", fontsize=12)
    ax.set_ylabel(r"Power-law Exponent ($\eta$)", fontsize=12)
    ax.set_title("XY: Correlation Decay Exponent", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid()
    fig.tight_layout()
    outpath = os.path.join(FIGURE_DIR, "xy_correlation_exponent.png")
    fig.savefig(outpath, dpi=FIGURE_DPI)
    print("Saved " + outpath)
    plt.close(fig)

#Probs leave this plot as discussion space is limited, eta fit shows
#this trend anyway just in a different way.
def plot_xy_correlation_length(filepaths):
    """
    Plot the exponential correlation length xi vs temperature.

    Fits C(r) ~ exp(-r/xi) at each temperature and plots xi.
    Above T_BKT xi is large just above the transition and decreases
    with increasing temperature as the disordered phase develops.

    Args:
        filepaths: List of .npz result file paths.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for res in _compute_fit_arrays(filepaths):
        ax.plot(res["temps"], res["xi_arr"],
                marker="o", markersize=3, label=res["label"])
    ax.set_xlabel("k_B T / J", fontsize=12)
    ax.set_ylabel("Correlation length xi (lattice units)", fontsize=12)
    ax.set_title("XY: exponential correlation length", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    outpath = os.path.join(FIGURE_DIR, "xy_correlation_length.png")
    fig.savefig(outpath, dpi=FIGURE_DPI)
    print("Saved " + outpath)
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Plot spin model simulation results")
    parser.add_argument("--ising", nargs="+", default=[], help="Ising .npz result files")
    parser.add_argument("--xy", nargs="+", default=[], help="XY .npz result files")
    return parser.parse_args()

def main():
    """Generate all figures from simulation result files."""
    args = parse_args()

    if args.ising:
        plot_ising(args.ising)
        plot_ising_magnetisation(args.ising)
    else:
        print("No Ising result files provided.")

    if args.xy:
        plot_xy(args.xy)
        plot_xy_correlations(args.xy)
        plot_xy_vortex_density(args.xy)
        plot_xy_correlation_exponent(args.xy)
        plot_xy_correlation_length(args.xy)
    else:
        print("No XY result files provided.")

    print("All figures saved to ./figures/")


if __name__ == "__main__":
    main()
