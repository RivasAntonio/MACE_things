#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MD mean analysis with automatic equilibration detection.

Reads an ASE-compatible trajectory and independently analyses selected
observables (chosen interactively by default):
  - Potential energy   [eV]
  - Total energy       [eV]   (potential + kinetic; falls back to E_pot if
                                momenta are absent)
  - Kinetic energy     [eV]   (skipped if momenta are absent)
  - Temperature        [K]    (skipped if momenta are absent)
  - Volume             [Ang^3]
  - Cell parameter A   [Ang]  (length of first lattice vector)
  - Cell parameter B   [Ang]
  - Cell parameter C   [Ang]
  - Pressure           [GPa]  (skipped if stress is absent)

For every observable a time-series plot and an autocorrelation plot are saved,
plus individual summary files and a single combined summary.

Usage
-----
    python traj_mdmean.py trajectory.traj          # interactive property selection
    python traj_mdmean.py trajectory.traj --all    # analyse all available properties
    python traj_mdmean.py OUTCAR --format vasp-out
    python traj_mdmean.py run.extxyz --index '::2'
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
from ase.io import read


# ==========================================================
# 0. CLI
# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="MD mean analysis on an ASE trajectory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "trajectory",
        nargs="?",
        default="trajectory.traj",
        help="Input trajectory (any ASE-readable format).",
    )
    parser.add_argument(
        "--format",
        default=None,
        metavar="FMT",
        help="ASE format string (auto-detected when omitted).",
    )
    parser.add_argument(
        "--index",
        default=":",
        help="Frame slice passed to ase.io.read (e.g. '::2', '100:').",
    )
    parser.add_argument(
        "--nblocks",
        type=int,
        default=7,
        help="Number of blocks to divide the production trajectory for SEM calculation.",
    )
    parser.add_argument(
        "--autocorr",
        action="store_true",
        help="Run standard autocorrelation analysis instead of block average analysis.",
    )
    parser.add_argument(
        "--no-fast",
        action="store_false",
        dest="fast",
        help="Do not use dynamic stride for faster equilibration detection.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_props",
        help="Analyse all available properties without prompting.",
    )
    args = parser.parse_args()
    if not args.autocorr and args.nblocks < 2:
        parser.error("nblocks must be at least 2 to compute SEM.")
    return args


# ==========================================================
# Property selection prompt
# ==========================================================
def prompt_property_selection(available_specs: list) -> list:
    """
    Interactively ask the user which properties to analyse.

    Parameters
    ----------
    available_specs : list of (key, label, units, tag)
        Only specs whose data array is not None.

    Returns
    -------
    list of (key, label, units, tag) chosen by the user.
    """
    print("\nAvailable properties:")
    for i, (key, label, units, tag) in enumerate(available_specs, start=1):
        unit_str = f" [{units}]" if units else ""
        print(f"  {i:2d}. {label}{unit_str}  (key: {key})")

    print()
    print("Enter the numbers of the properties you want to analyse,")
    print("separated by spaces or commas (e.g. '1 3 5' or '1,3,5').")
    print("Press Enter with no input to select ALL properties.")

    while True:
        raw = input("> ").strip()
        if not raw:
            print("No selection made — analysing ALL available properties.")
            return list(available_specs)
        # Accept both comma- and space-separated lists
        tokens = raw.replace(",", " ").split()
        try:
            indices = [int(t) for t in tokens]
        except ValueError:
            print("Invalid input. Please enter numbers only.")
            continue
        invalid = [i for i in indices if i < 1 or i > len(available_specs)]
        if invalid:
            print(f"Numbers out of range: {invalid}. Valid range: 1–{len(available_specs)}.")
            continue
        selected = [available_specs[i - 1] for i in indices]
        labels = ", ".join(s[1] for s in selected)
        print(f"Selected: {labels}")
        return selected


# ==========================================================
# 1. FFT autocorrelation
# ==========================================================
def autocorrelation_fft(x: np.ndarray) -> np.ndarray:
    """Normalized autocorrelation C(k) via FFT (O(N log N))."""
    x = x - x.mean()
    N = len(x)
    f = fft(x, n=2 * N)
    acf = ifft(f * np.conj(f))[:N].real
    if acf[0] == 0.0:
        return np.zeros(N)
    acf /= acf[0]
    return acf


def integrated_autocorr_time(corr: np.ndarray) -> float:
    """
    tau_int = 1 + 2 * sum_{k>=1} C(k).
    Sum is truncated at the first negative C(k) (initial positive sequence).
    """
    tau = 1.0
    for k in range(1, len(corr)):
        if corr[k] < 0.0:
            break
        tau += 2.0 * corr[k]
    return tau


# ==========================================================
# 2. Equilibration detection
# ==========================================================
def detect_equilibration(series: np.ndarray, fast: bool = False) -> tuple[int, float]:
    """
    Find t0 in [0, N/2) that maximises N_eff = (N-t0)/tau_int.
    If fast is True, uses a dynamic stride to ensure fast evaluation (at most ~200 checks).
    Otherwise uses a stride of 5.
    """
    N = len(series)
    best_t0, best_Neff = 0, 0.0
    stride = max(5, N // 200) if fast else 5
    for t0 in range(0, N // 2, stride):
        truncated = series[t0:]
        corr = autocorrelation_fft(truncated)
        tau = integrated_autocorr_time(corr)
        Neff = len(truncated) / tau
        if Neff > best_Neff:
            best_Neff = Neff
            best_t0 = t0
    return best_t0, best_Neff


def plot_running_mean(
    series: np.ndarray,
    t0: int,
    mean_val: float,
    SEM: float,
    label: str,
    units: str,
    tag: str,
    output_dir: Path,
):
    """Plot the evolution of the running mean over the trajectory."""
    N = len(series)
    N_prod = N - t0
    frames_idx = np.arange(N)
    frames_prod = frames_idx[t0:]
    production = series[t0:]
    unit_str = f" {units}" if units else ""

    fig, ax = plt.subplots(figsize=(12, 5))

    # 1. Running mean from start
    run_mean_start = np.cumsum(series) / np.arange(1, N + 1)
    ax.plot(frames_idx, run_mean_start, lw=1.5, color="coral", label="Running Mean (from start)")

    # 2. Running mean from t0 (production)
    if t0 < N:
        run_mean_prod = np.cumsum(production) / np.arange(1, N_prod + 1)
        ax.plot(frames_prod, run_mean_prod, lw=2.0, color="navy", label=f"Running Mean (from t0={t0})")

    # 3. Overall mean line and SEM band
    ax.axhline(mean_val, color="navy", ls="--", lw=1.2, label=f"Final Mean = {mean_val:.6g}")
    ax.fill_between(
        frames_prod,
        mean_val - SEM,
        mean_val + SEM,
        color="navy",
        alpha=0.15,
        label=f"SEM = {SEM:.4g}{unit_str}",
    )

    # 4. Vertical line at t0
    if t0 > 0:
        ax.axvline(t0, color="red", ls=":", lw=1.5, label=f"t0 = {t0}")
        ax.axvspan(0, t0, color="lightcoral", alpha=0.08, label="Equilibration Phase")

    ax.set_title(f"{label}: Evolution of the Mean")
    ax.set_xlabel("Frame index")
    ax.set_ylabel(f"Running Mean{unit_str}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / f"running_mean_{tag}.png", dpi=200)
    plt.close(fig)


# ==========================================================
# 3. Per-observable analysis + plots
# ==========================================================
def analyse(
    series: np.ndarray,
    label: str,
    units: str,
    tag: str,
    output_dir: Path,
    nblocks: int | None = None,
    fast: bool = False,
) -> dict:
    """
    Run equilibration detection, compute statistics, and save plots and
    a summary file for a single observable.
    Supports either standard autocorrelation analysis or block average analysis.
    """
    N = len(series)
    t0, _ = detect_equilibration(series, fast=fast)
    production = series[t0:]
    N_prod = len(production)

    if nblocks is not None:
        if N_prod < nblocks:
            raise ValueError(
                f"Production series length ({N_prod}) after equilibration t0={t0} is smaller "
                f"than the requested number of blocks ({nblocks})."
            )

        block_size = N_prod // nblocks
        discarded = N_prod % nblocks
        used_len = nblocks * block_size
        production_used = production[:used_len]

        # Compute block averages
        block_means = np.zeros(nblocks)
        for b in range(nblocks):
            block_means[b] = np.mean(production_used[b * block_size : (b + 1) * block_size])

        # Calculate statistics
        mean_val = np.mean(block_means)
        block_sd = np.std(block_means, ddof=1)
        SEM = block_sd / np.sqrt(nblocks)
        raw_sd = np.std(production_used, ddof=1)

        # Compute autocorrelation for plot and summary statistics
        corr = autocorrelation_fft(production)
        tau_int = integrated_autocorr_time(corr)
        Neff = len(production) / tau_int

        # ---- summary text ------------------------------------------------
        lines = [
            f"===== {label.upper()} =====",
            f"Units                      = {units}",
            f"Total frames               = {N}",
            f"Equilibration t0           = {t0}",
            f"Production length          = {N_prod}",
            f"Number of blocks           = {nblocks}",
            f"Block size                 = {block_size}",
            f"Leftover production frames = {discarded}",
            f"Mean                       = {mean_val:.6f}",
            f"Std Dev (raw production)   = {raw_sd:.6f}",
            f"Block SD                   = {block_sd:.6f}",
            f"SEM                        = {SEM:.6f}",
            f"Integrated tau_int         = {tau_int:.6f}",
            f"Effective sample size Neff = {Neff:.3f}",
        ]
        summary = "\n".join(lines) + "\n"
        print(summary)
        (output_dir / f"summary_{tag}.txt").write_text(summary)

        # ---- time-series plot --------------------------------------------
        frames_idx = np.arange(N)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(frames_idx, series, lw=1, color="steelblue", label=label)

        # Shade equilibration region
        if t0 > 0:
            ax.axvspan(0, t0, color="lightcoral", alpha=0.15, label=f"Equilibration (t0={t0})")
            ax.axvline(t0, color="red", ls="--", lw=1.5)

        # Shade discarded region if any
        if discarded > 0:
            ax.axvspan(t0 + used_len, N - 1, color="lightgray", alpha=0.3, label="Leftover Production (end)")

        # Plot vertical block boundary lines and horizontal block means
        for b in range(nblocks):
            x_start = t0 + b * block_size
            x_end = t0 + (b + 1) * block_size - 1
            if b > 0:
                ax.axvline(x_start, color="gray", ls=":", lw=1)
            ax.hlines(block_means[b], x_start, x_end, color="darkorange", lw=2,
                      label="Block Means" if b == 0 else "")

        ax.axhline(mean_val, color="navy", ls="-", lw=1.8, label="Overall Mean")

        ax.text(
            t0 + 0.02 * (N - t0) if t0 < N - 1 else 0, mean_val,
            f"Mean = {mean_val:.5g} {units}",
            color="navy", fontsize=9, va="bottom",
        )

        ax.fill_between(
            frames_idx[t0 : t0 + used_len],
            mean_val - SEM,
            mean_val + SEM,
            color="gray", alpha=0.25, label=f"SEM = {SEM:.4g} {units}",
        )

        ax.set_title(f"{label}: Fast Block Analysis after Equilibration ({nblocks} Blocks, size {block_size})")
        ax.set_xlabel("Frame index")
        ax.set_ylabel(f"{label} [{units}]")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(output_dir / f"timeseries_{tag}.png", dpi=200)
        plt.close(fig)

        # ---- autocorrelation plot ----------------------------------------
        plot_len = min(500, len(corr))
        kmax = next(
            (k for k in range(1, len(corr)) if corr[k] < 0.0),
            plot_len - 1,
        )

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(corr[:plot_len], lw=1.5, color="steelblue", label="C(k)")
        ax.axhline(0.0, color="black", lw=1, ls="--")
        ax.axvline(kmax, color="red", ls="--", lw=1.5, label=f"k_max = {kmax}")
        ax.set_title(f"{label}: Normalized Autocorrelation C(k)")
        ax.set_xlabel("Lag k")
        ax.set_ylabel("C(k)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(output_dir / f"autocorrelation_{tag}.png", dpi=200)
        plt.close(fig)

        plot_running_mean(
            series=series,
            t0=t0,
            mean_val=mean_val,
            SEM=SEM,
            label=label,
            units=units,
            tag=tag,
            output_dir=output_dir,
        )

        return dict(
            label=label,
            t0=t0,
            nblocks=nblocks,
            block_size=block_size,
            discarded=discarded,
            mean=mean_val,
            raw_sd=raw_sd,
            block_sd=block_sd,
            SEM=SEM,
            tau_int=tau_int,
            Neff=Neff,
        )

    else:
        # Standard autocorrelation analysis
        sigma   = np.std(production, ddof=1)
        corr    = autocorrelation_fft(production)
        tau_int = integrated_autocorr_time(corr)
        Neff    = len(production) / tau_int
        mean_val = np.mean(production)
        SEM     = sigma / np.sqrt(Neff)

        # ---- summary text ------------------------------------------------
        lines = [
            f"===== {label.upper()} =====",
            f"Units                      = {units}",
            f"Total frames               = {N}",
            f"Equilibration t0           = {t0}",
            f"Production length          = {len(production)}",
            f"Integrated tau_int         = {tau_int:.6f}",
            f"Effective sample size Neff = {Neff:.3f}",
            f"Mean (production)          = {mean_val:.6f}",
            f"Std Dev (production)       = {sigma:.6f}",
            f"SEM                        = {SEM:.6f}",
        ]
        summary = "\n".join(lines) + "\n"
        print(summary)
        (output_dir / f"summary_{tag}.txt").write_text(summary)

        # ---- time-series plot --------------------------------------------
        frames_idx = np.arange(N)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(frames_idx, series, lw=1, color="steelblue", label=label)
        ax.axvline(t0, color="red", ls="--", lw=1.5, label=f"t0 = {t0}")
        ax.axhline(mean_val, color="navy", ls="-", lw=1.8)
        ax.text(
            0.02 * N, mean_val,
            f"Mean = {mean_val:.5g} {units}",
            color="navy", fontsize=9, va="bottom",
        )
        ax.fill_between(
            frames_idx,
            mean_val - SEM,
            mean_val + SEM,
            color="gray", alpha=0.25, label=f"SEM = {SEM:.4g} {units}",
        )
        # tau_int visualised as a horizontal bar placed safely above the data
        y_tau  = mean_val + 2.0 * sigma
        x0_tau = int(0.70 * N)
        ax.hlines(y_tau, x0_tau, x0_tau + tau_int, color="green", lw=3)
        ax.text(
            x0_tau, y_tau,
            f"tau_int = {tau_int:.1f}",
            color="green", fontsize=9, va="bottom",
        )
        ax.set_title(f"{label}: Equilibration Detection, Mean, SEM, tau_int")
        ax.set_xlabel("Frame index")
        ax.set_ylabel(f"{label} [{units}]")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(output_dir / f"timeseries_{tag}.png", dpi=200)
        plt.close(fig)

        # ---- autocorrelation plot ----------------------------------------
        plot_len = min(500, len(corr))
        kmax = next(
            (k for k in range(1, len(corr)) if corr[k] < 0.0),
            plot_len - 1,
        )

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(corr[:plot_len], lw=1.5, color="steelblue", label="C(k)")
        ax.axhline(0.0, color="black", lw=1, ls="--")
        ax.axvline(kmax, color="red", ls="--", lw=1.5, label=f"k_max = {kmax}")
        ax.set_title(f"{label}: Normalized Autocorrelation C(k)")
        ax.set_xlabel("Lag k")
        ax.set_ylabel("C(k)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(output_dir / f"autocorrelation_{tag}.png", dpi=200)
        plt.close(fig)

        plot_running_mean(
            series=series,
            t0=t0,
            mean_val=mean_val,
            SEM=SEM,
            label=label,
            units=units,
            tag=tag,
            output_dir=output_dir,
        )

        return dict(
            label=label, t0=t0, mean=mean_val, sd=sigma, SEM=SEM, tau_int=tau_int, Neff=Neff
        )


# ==========================================================
# 4. Trajectory loading and observable extraction
# ==========================================================
def load_trajectory(path: Path, index: str, fmt: str | None):
    """Return a list of ASE Atoms objects."""
    print(f"Reading trajectory: {path}")
    frames = read(str(path), index=index, format=fmt)
    if not isinstance(frames, list):
        frames = [frames]
    print(f"  Loaded {len(frames)} frames.")
    return frames


def extract_observables(frames: list) -> dict[str, np.ndarray | None]:
    """
    Extract per-frame arrays for all target observables.

    Returns a dict:
        epot   : potential energy [eV]
        etot   : total energy [eV]  (None if momenta absent)
        ekin   : kinetic energy [eV] (None if momenta absent)
        temp   : temperature [K] (None if momenta absent)
        vol    : volume [Ang^3]
        cell_a : first cell vector length [Ang]
        cell_b : second cell vector length [Ang]
        cell_c : third cell vector length [Ang]
        pres   : pressure [GPa] (None if stress absent)
    """
    # Check whether momenta (velocities) are stored in the first frame.
    # ASE stores momenta as p = m*v; Atoms.has('momenta') is the canonical check.
    has_momenta = frames[0].has("momenta")
    if not has_momenta:
        warnings.warn(
            "Trajectory does not contain momenta/velocities. "
            "Total energy will equal potential energy, "
            "Kinetic energy and Temperature analyses will be skipped.",
            stacklevel=2,
        )

    # Check whether stress info is stored/calculable in the first frame.
    has_stress = False
    try:
        frames[0].get_stress()
        has_stress = True
    except Exception:
        warnings.warn(
            "Trajectory does not contain stress information. "
            "Pressure analysis will be skipped.",
            stacklevel=2,
        )

    epot   = np.empty(len(frames))
    etot   = np.empty(len(frames)) if has_momenta else None
    ekin   = np.empty(len(frames)) if has_momenta else None
    temp   = np.empty(len(frames)) if has_momenta else None
    vol    = np.empty(len(frames))
    cell_a = np.empty(len(frames))
    cell_b = np.empty(len(frames))
    cell_c = np.empty(len(frames))
    pres   = np.empty(len(frames)) if has_stress else None

    EV_A3_TO_GPA = 160.21766

    for i, atoms in enumerate(frames):
        epot[i]   = atoms.get_potential_energy()
        vol[i]    = atoms.get_volume()
        cell_lengths = atoms.cell.lengths()
        cell_a[i] = cell_lengths[0]
        cell_b[i] = cell_lengths[1]
        cell_c[i] = cell_lengths[2]

        if has_momenta:
            ekin[i] = atoms.get_kinetic_energy()
            etot[i] = epot[i] + ekin[i]
            temp[i] = atoms.get_temperature()

        if has_stress:
            s = atoms.get_stress()
            # Stress in Voigt notation: [xx, yy, zz, yz, xz, xy]
            pres[i] = - (s[0] + s[1] + s[2]) / 3.0 * EV_A3_TO_GPA

    return dict(epot=epot, etot=etot, ekin=ekin, temp=temp, vol=vol, cell_a=cell_a, cell_b=cell_b, cell_c=cell_c, pres=pres)


# ==========================================================
# 5. Main
# ==========================================================
def main():
    args = parse_args()
    traj_path = Path(args.trajectory)

    if not traj_path.exists():
        raise FileNotFoundError(f"Input trajectory file not found: {traj_path}")

    # Determine analysis type
    nblocks = None if args.autocorr else args.nblocks

    # Output folder named after the trajectory file, created next to it.
    if nblocks is not None:
        if args.fast:
            dir_name = f"fast_block_mdmean_{traj_path.stem}"
        else:
            dir_name = f"block_mdmean_{traj_path.stem}"
    else:
        dir_name = f"mdmean_{traj_path.stem}"
    output_dir = traj_path.parent / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")

    frames = load_trajectory(traj_path, index=args.index, fmt=args.format)
    obs = extract_observables(frames)

    # Define the observables to analyse in order.
    # Each entry: (array_key, label, units, filename_tag)
    all_specs = [
        ("epot",   "Potential Energy", "eV",     "epot"),
        ("etot",   "Total Energy",     "eV",     "etot"),
        ("ekin",   "Kinetic Energy",   "eV",     "ekin"),
        ("temp",   "Temperature",      "K",      "temp"),
        ("vol",    "Volume",           "Ang^3",  "vol"),
        ("cell_a", "Cell Parameter A", "Ang",    "cell_a"),
        ("cell_b", "Cell Parameter B", "Ang",    "cell_b"),
        ("cell_c", "Cell Parameter C", "Ang",    "cell_c"),
        ("pres",   "Pressure",         "GPa",    "pres"),
    ]

    # Filter to only specs with data available
    available_specs = [
        (key, label, units, tag)
        for key, label, units, tag in all_specs
        if obs[key] is not None
    ]

    unavailable = [
        label
        for key, label, units, tag in all_specs
        if obs[key] is None
    ]
    if unavailable:
        print(f"  The following properties are unavailable (data missing): {', '.join(unavailable)}")

    # Select which properties to analyse
    if args.all_props:
        selected_specs = available_specs
        print(f"\n--all flag set: analysing all {len(selected_specs)} available properties.")
    else:
        selected_specs = prompt_property_selection(available_specs)

    results = []
    for key, label, units, tag in selected_specs:
        series = obs[key]
        results.append(analyse(series, label, units, tag, output_dir, nblocks, args.fast))

    # ---- combined summary --------------------------------------------
    header = "===== COMBINED SUMMARY =====\n"
    if nblocks is not None:
        col_w = max(len(r["label"]) for r in results) + 2
        rows = [
            f"{'Observable':<{col_w}}  {'t0':>6}  {'Blocks':>6}  {'BlockSize':>9}  {'Leftover':>9}  "
            f"{'Mean':>16}  {'SD (raw)':>12}  {'Block SD':>12}  {'SEM':>12}  {'tau_int':>10}  {'Neff':>8}",
        ]
        separator = "-" * len(rows[0])
        rows.append(separator)
        for r in results:
            rows.append(
                f"{r['label']:<{col_w}}  {r['t0']:>6d}  {r['nblocks']:>6d}  {r['block_size']:>9d}  {r['discarded']:>9d}  "
                f"{r['mean']:>16.6f}  {r['raw_sd']:>12.6f}  {r['block_sd']:>12.6f}  {r['SEM']:>12.6f}  {r['tau_int']:>10.2f}  {r['Neff']:>8.1f}"
            )
    else:
        col_w = max(len(r["label"]) for r in results) + 2
        rows = [
            f"{'Observable':<{col_w}}  {'t0':>6}  {'Mean':>16}  "
            f"{'SD':>12}  {'SEM':>12}  {'tau_int':>10}  {'Neff':>8}",
            "-" * (col_w + 76),
        ]
        for r in results:
            rows.append(
                f"{r['label']:<{col_w}}  {r['t0']:>6d}  {r['mean']:>16.6f}  "
                f"{r['sd']:>12.6f}  {r['SEM']:>12.6f}  {r['tau_int']:>10.2f}  {r['Neff']:>8.1f}"
            )
    combined = header + "\n".join(rows) + "\n"
    print(combined)
    (output_dir / "summary_combined.txt").write_text(combined)
    try:
        parent_dir = traj_path.resolve().parent
        if parent_dir.name == "outputs":
            outputs_dir = parent_dir
        else:
            outputs_dir = parent_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        combined_path = outputs_dir / "summary_combined.txt"
        combined_path.write_text(combined)
        print(f"Combined summary written to: {combined_path.resolve()}")
    except Exception as e:
        print(f"Could not write combined summary to outputs folder: {e}")


if __name__ == "__main__":
    main()
