#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MD mean analysis with automatic equilibration detection on text-based log files.

Reads a text-based log file and independently analyses each column
except the first one (typically Time). By default the user is prompted
to select which properties to analyse; use --all to process every column.

For every observable a time-series plot and an autocorrelation plot are saved,
plus individual summary files and a single combined summary.

Usage
-----
    python log_mdmean.py md.log                # interactive property selection
    python log_mdmean.py md.log --all          # analyse all columns
    python log_mdmean.py md.log --index '100:'
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft


# ==========================================================
# 0. CLI and Helper Parsers
# ==========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="MD mean analysis on a log file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        default="md.log",
        help="Input log file (default: md.log).",
    )
    parser.add_argument(
        "--index",
        default=":",
        help="Slice string to subset the rows (e.g. '::2', '100:', '-500:').",
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
        help="Analyse all available columns without prompting.",
    )
    args = parser.parse_args()
    if not args.autocorr and args.nblocks < 2:
        parser.error("nblocks must be at least 2 to compute SEM.")
    return args


# ==========================================================
# Property selection prompt
# ==========================================================
def prompt_property_selection(col_specs: list) -> list:
    """
    Interactively ask the user which columns to analyse.

    Parameters
    ----------
    col_specs : list of (col_idx, header, label, units, tag)

    Returns
    -------
    list of selected (col_idx, header, label, units, tag).
    """
    print("\nAvailable properties:")
    for i, (col_idx, header, label, units, tag) in enumerate(col_specs, start=1):
        unit_str = f" [{units}]" if units else ""
        print(f"  {i:2d}. {label}{unit_str}  (column: {header})")

    print()
    print("Enter the numbers of the properties you want to analyse,")
    print("separated by spaces or commas (e.g. '1 3 5' or '1,3,5').")
    print("Press Enter with no input to select ALL properties.")

    while True:
        raw = input("> ").strip()
        if not raw:
            print("No selection made — analysing ALL available properties.")
            return list(col_specs)
        tokens = raw.replace(",", " ").split()
        try:
            indices = [int(t) for t in tokens]
        except ValueError:
            print("Invalid input. Please enter numbers only.")
            continue
        invalid = [i for i in indices if i < 1 or i > len(col_specs)]
        if invalid:
            print(f"Numbers out of range: {invalid}. Valid range: 1–{len(col_specs)}.")
            continue
        selected = [col_specs[i - 1] for i in indices]
        labels = ", ".join(s[2] for s in selected)
        print(f"Selected: {labels}")
        return selected



def parse_index(index_str: str) -> slice:
    """Parse index string into a slice object."""
    index_str = index_str.strip()
    if not index_str or index_str == ":":
        return slice(None)
    if ":" in index_str:
        parts = index_str.split(":")
        # Pad with None if fewer than 3 parts
        parts += [None] * (3 - len(parts))
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if parts[1] else None
        step = int(parts[2]) if parts[2] else None
        return slice(start, stop, step)
    else:
        # If it's a single integer, treat as starting index of a slice (e.g., "100" -> slice(100, None))
        val = int(index_str)
        return slice(val, None)


def parse_header(header: str) -> tuple[str, str, str]:
    """
    Parse a column header to extract the name, units, and a clean tag.
    Example: "Etot[eV]" -> ("Total Energy", "eV", "etot")
    """
    # Find patterns like Name[Units]
    match = re.match(r"^([^\[\s]+)(?:\[([^\]]+)\])?$", header)
    if match:
        name = match.group(1).strip()
        units = match.group(2).strip() if match.group(2) else ""
    else:
        name = header
        units = ""

    # Map common MD variable abbreviations to user-friendly titles and tags
    mappings = {
        "Etot": ("Total Energy", "etot"),
        "Epot": ("Potential Energy", "epot"),
        "Ekin": ("Kinetic Energy", "ekin"),
        "T": ("Temperature", "temp"),
        "Temp": ("Temperature", "temp"),
        "Temperature": ("Temperature", "temp"),
        "Volume": ("Volume", "vol"),
        "Vol": ("Volume", "vol"),
        "Pres": ("Pressure", "pres"),
        "Press": ("Pressure", "pres"),
        "P": ("Pressure", "pres"),
        "Density": ("Density", "density"),
        "Dens": ("Density", "density"),
    }

    clean_name = re.sub(r"[^\w\s-]", "", name).strip()

    # Case-insensitive match in mappings
    if name in mappings:
        label, tag = mappings[name]
    elif name.lower() in {k.lower(): v for k, v in mappings.items()}:
        lower_map = {k.lower(): v for k, v in mappings.items()}
        label, tag = lower_map[name.lower()]
    else:
        label = clean_name
        tag = clean_name.lower().replace(" ", "_")

    return label, units, tag


# ==========================================================
# 1. FFT Autocorrelation
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
# 2. Equilibration Detection
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
    times: np.ndarray,
    t0: int,
    mean_val: float,
    SEM: float,
    label: str,
    units: str,
    tag: str,
    time_label: str,
    time_units: str,
    output_dir: Path,
):
    """Plot the evolution of the running mean over the trajectory."""
    N = len(series)
    N_prod = N - t0
    times_prod = times[t0:]
    production = series[t0:]
    unit_str = f" {units}" if units else ""

    fig, ax = plt.subplots(figsize=(12, 5))

    # 1. Running mean from start
    run_mean_start = np.cumsum(series) / np.arange(1, N + 1)
    ax.plot(times, run_mean_start, lw=1.5, color="coral", label="Running Mean (from start)")

    # 2. Running mean from t0 (production)
    if t0 < N:
        run_mean_prod = np.cumsum(production) / np.arange(1, N_prod + 1)
        ax.plot(times_prod, run_mean_prod, lw=2.0, color="navy", label=f"Running Mean (from t0={t0})")

    # 3. Overall mean line and SEM band
    ax.axhline(mean_val, color="navy", ls="--", lw=1.2, label=f"Final Mean = {mean_val:.6g}")
    ax.fill_between(
        times_prod,
        mean_val - SEM,
        mean_val + SEM,
        color="navy",
        alpha=0.15,
        label=f"SEM = {SEM:.4g}{unit_str}",
    )

    # 4. Vertical line at t0
    if t0 > 0:
        ax.axvline(times[t0], color="red", ls=":", lw=1.5, label=f"t0 = {t0}")
        ax.axvspan(times[0], times[t0], color="lightcoral", alpha=0.08, label="Equilibration Phase")

    ax.set_title(f"{label}: Evolution of the Mean")
    ax.set_xlabel(f"{time_label}{' [' + time_units + ']' if time_units else ''}")
    ax.set_ylabel(f"Running Mean{unit_str}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / f"running_mean_{tag}.png", dpi=200)
    plt.close(fig)


# ==========================================================
# 3. Per-Observable Analysis + Plots
# ==========================================================
def analyse(
    series: np.ndarray,
    label: str,
    units: str,
    tag: str,
    times: np.ndarray,
    time_label: str,
    time_units: str,
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
    times_prod = times[t0:]
    N_prod = len(production)

    time_unit_str = f" {time_units}" if time_units else ""
    dt = (times[-1] - times[0]) / (N - 1) if N > 1 else 1.0

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
        times_prod_used = times_prod[:used_len]

        # Calculate block averages
        block_means = np.zeros(nblocks)
        for b in range(nblocks):
            block_means[b] = np.mean(production_used[b * block_size : (b + 1) * block_size])

        # Calculate statistics
        mean_val = np.mean(block_means)
        block_sd = np.std(block_means, ddof=1)
        SEM = block_sd / np.sqrt(nblocks)
        raw_sd = np.std(production_used, ddof=1)
        block_time = block_size * dt

        # Compute autocorrelation for plot and summary statistics
        corr = autocorrelation_fft(production)
        tau_int = integrated_autocorr_time(corr)
        Neff = len(production) / tau_int
        tau_int_time = tau_int * dt

        # ---- summary text ------------------------------------------------
        lines = [
            f"===== {label.upper()} =====",
            f"Units                      = {units}",
            f"Total frames               = {N}",
            f"Equilibration t0 (index)   = {t0}",
            f"Equilibration t0 (time)    = {times[t0]:.4f}{time_unit_str}",
            f"Production length          = {N_prod}",
            f"Number of blocks           = {nblocks}",
            f"Block size (frames)        = {block_size}",
            f"Block size (time)          = {block_time:.6f}{time_unit_str}",
            f"Leftover production frames = {discarded}",
            f"Mean (block-averaged)      = {mean_val:.6f}",
            f"Std Dev (raw production)   = {raw_sd:.6f}",
            f"Block SD                   = {block_sd:.6f}",
            f"SEM                        = {SEM:.6f}",
            f"Integrated tau_int (steps) = {tau_int:.6f}",
            f"Integrated tau_int (time)  = {tau_int_time:.6f}{time_unit_str}",
            f"Effective sample size Neff = {Neff:.3f}",
        ]
        summary = "\n".join(lines) + "\n"
        print(summary)
        (output_dir / f"summary_{tag}.txt").write_text(summary)

        # ---- time-series plot --------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(times, series, lw=1, color="steelblue", label=label)

        # Shade equilibration region
        if t0 > 0:
            ax.axvspan(times[0], times[t0], color="lightcoral", alpha=0.15, label=f"Equilibration (t0={t0})")
            ax.axvline(times[t0], color="red", ls="--", lw=1.5)

        # Shade discarded region if any
        if discarded > 0:
            discard_start_time = times[t0 + used_len]
            ax.axvspan(discard_start_time, times[-1], color="lightgray", alpha=0.3, label="Leftover Production (end)")

        # Plot vertical block boundary lines and horizontal block means
        for b in range(nblocks):
            t_start = times_prod_used[b * block_size]
            t_end = times_prod_used[min((b + 1) * block_size - 1, used_len - 1)]
            if b > 0:
                ax.axvline(t_start, color="gray", ls=":", lw=1)
            ax.hlines(block_means[b], t_start, t_end, color="darkorange", lw=2,
                      label="Block Means" if b == 0 else "")

        # Plot overall mean line
        ax.axhline(mean_val, color="navy", ls="-", lw=1.8, label="Overall Mean")

        unit_str = f" {units}" if units else ""
        ax.text(
            times[t0] + 0.02 * (times[-1] - times[t0]) if t0 < N - 1 else times[0],
            mean_val,
            f"Mean = {mean_val:.5g}{unit_str}",
            color="navy",
            fontsize=9,
            va="bottom",
        )

        # SEM shaded region
        ax.fill_between(
            times_prod_used,
            mean_val - SEM,
            mean_val + SEM,
            color="gray",
            alpha=0.25,
            label=f"SEM = {SEM:.4g}{unit_str}",
        )

        ax.set_title(f"{label}: Fast Block Analysis after Equilibration ({nblocks} Blocks, size {block_size})")
        ax.set_xlabel(f"{time_label}{' [' + time_units + ']' if time_units else ''}")
        ax.set_ylabel(f"{label}{' [' + units + ']' if units else ''}")
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
            times=times,
            t0=t0,
            mean_val=mean_val,
            SEM=SEM,
            label=label,
            units=units,
            tag=tag,
            time_label=time_label,
            time_units=time_units,
            output_dir=output_dir,
        )

        return dict(
            label=label,
            t0=t0,
            t0_time=times[t0],
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
        sigma = np.std(production, ddof=1)
        corr = autocorrelation_fft(production)
        tau_int = integrated_autocorr_time(corr)
        Neff = len(production) / tau_int
        mean_val = np.mean(production)
        SEM = sigma / np.sqrt(Neff)
        tau_int_time = tau_int * dt

        # ---- summary text ------------------------------------------------
        lines = [
            f"===== {label.upper()} =====",
            f"Units                      = {units}",
            f"Total frames               = {N}",
            f"Equilibration t0 (index)   = {t0}",
            f"Equilibration t0 (time)    = {times[t0]:.4f}{time_unit_str}",
            f"Production length          = {len(production)}",
            f"Integrated tau_int (steps) = {tau_int:.6f}",
            f"Integrated tau_int (time)  = {tau_int_time:.6f}{time_unit_str}",
            f"Effective sample size Neff = {Neff:.3f}",
            f"Mean (production)          = {mean_val:.6f}",
            f"Std Dev (production)       = {sigma:.6f}",
            f"SEM                        = {SEM:.6f}",
        ]
        summary = "\n".join(lines) + "\n"
        print(summary)
        (output_dir / f"summary_{tag}.txt").write_text(summary)

        # ---- time-series plot --------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(times, series, lw=1, color="steelblue", label=label)

        # Vertical line at equilibration time
        eq_time = times[t0]
        ax.axvline(eq_time, color="red", ls="--", lw=1.5, label=f"t0 = {t0} ({eq_time:.2f}{time_unit_str})")

        ax.axhline(mean_val, color="navy", ls="-", lw=1.8)

        unit_str = f" {units}" if units else ""
        ax.text(
            times[0] + 0.02 * (times[-1] - times[0]),
            mean_val,
            f"Mean = {mean_val:.5g}{unit_str}",
            color="navy",
            fontsize=9,
            va="bottom",
        )
        ax.fill_between(
            times,
            mean_val - SEM,
            mean_val + SEM,
            color="gray",
            alpha=0.25,
            label=f"SEM = {SEM:.4g}{unit_str}",
        )

        # tau_int visualised as a horizontal bar placed safely above the data
        y_tau = mean_val + 2.0 * sigma
        x0_tau = times[int(0.70 * N)]
        ax.hlines(y_tau, x0_tau, x0_tau + tau_int_time, color="green", lw=3)
        ax.text(
            x0_tau,
            y_tau,
            f"tau_int = {tau_int:.1f} ({tau_int_time:.2f}{time_unit_str})",
            color="green",
            fontsize=9,
            va="bottom",
        )

        ax.set_title(f"{label}: Equilibration Detection, Mean, SEM, tau_int")
        ax.set_xlabel(f"{time_label}{' [' + time_units + ']' if time_units else ''}")
        ax.set_ylabel(f"{label}{' [' + units + ']' if units else ''}")
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
            times=times,
            t0=t0,
            mean_val=mean_val,
            SEM=SEM,
            label=label,
            units=units,
            tag=tag,
            time_label=time_label,
            time_units=time_units,
            output_dir=output_dir,
        )

        return dict(
            label=label,
            t0=t0,
            t0_time=eq_time,
            mean=mean_val,
            sd=sigma,
            SEM=SEM,
            tau_int=tau_int,
            Neff=Neff,
        )


# ==========================================================
# 4. Log loading and parsing
# ==========================================================
def load_log_file(path: Path, index: str) -> tuple[list[str], np.ndarray]:
    """
    Read a log file and return headers and data array.
    """
    print(f"Reading log file: {path}")

    header_line = None
    with open(path, "r") as f:
        for line in f:
            cleaned = line.strip()
            if cleaned:
                header_line = cleaned
                break

    if not header_line:
        raise ValueError(f"Log file {path} is empty or has no content.")

    # Strip potential comment characters from the start of the header
    comment_chars = "#"
    while header_line and header_line[0] in comment_chars:
        header_line = header_line[1:].strip()

    headers = header_line.split()
    if not headers:
        raise ValueError(f"Could not parse any headers from first non-empty line: '{header_line}'")

    # Load numeric data, ignoring comment lines starting with '#'
    data = np.loadtxt(path, skiprows=1, comments="#")

    if data.ndim == 1:
        if len(headers) == 1:
            data = np.expand_dims(data, axis=1)
        else:
            if len(data) == len(headers):
                data = np.expand_dims(data, axis=0)
            else:
                data = np.expand_dims(data, axis=1)

    if data.shape[1] != len(headers):
        raise ValueError(
            f"Number of columns in data ({data.shape[1]}) does not match "
            f"number of headers ({len(headers)})."
        )

    # Slice data
    idx = parse_index(index)
    data = data[idx]
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)

    print(f"  Loaded {len(data)} data points for {len(headers)} columns.")
    return headers, data


# ==========================================================
# 5. Main
# ==========================================================
def main():
    args = parse_args()
    log_path = Path(args.log_file)

    if not log_path.exists():
        raise FileNotFoundError(f"Input log file not found: {log_path}")

    # Determine analysis type
    nblocks = None if args.autocorr else args.nblocks

    # Output folder named after the log file, created next to it.
    if nblocks is not None:
        if args.fast:
            dir_name = f"fast_block_mdmean_{log_path.stem}"
        else:
            dir_name = f"block_mdmean_{log_path.stem}"
    else:
        dir_name = f"mdmean_{log_path.stem}"
    output_dir = log_path.parent / dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")

    headers, data = load_log_file(log_path, index=args.index)

    if len(headers) < 2:
        raise ValueError(
            f"Log file must have at least 2 columns (e.g. Time and an observable). "
            f"Found headers: {headers}"
        )

    # First column is typically Time or Step (skip analysis, use for X axis)
    time_col_header = headers[0]
    time_label, time_units, _ = parse_header(time_col_header)
    times = data[:, 0]

    results = []
    # Build specs list for all analysable columns (skip first/time column)
    col_specs = []
    for col_idx in range(1, len(headers)):
        header = headers[col_idx]
        label, units, tag = parse_header(header)
        col_specs.append((col_idx, header, label, units, tag))

    # Select which columns to analyse
    if args.all_props:
        selected_specs = col_specs
        print(f"\n--all flag set: analysing all {len(selected_specs)} available properties.")
    else:
        selected_specs = prompt_property_selection(col_specs)

    # Analyse selected columns
    for col_idx, header, label, units, tag in selected_specs:
        series = data[:, col_idx]
        print(f"\nAnalyzing column {col_idx}: {header} ({label})")
        res = analyse(
            series=series,
            label=label,
            units=units,
            tag=tag,
            times=times,
            time_label=time_label,
            time_units=time_units,
            output_dir=output_dir,
            nblocks=nblocks,
            fast=args.fast,
        )
        results.append(res)

    # ---- combined summary --------------------------------------------
    header_text = "===== COMBINED SUMMARY =====\n"
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
            f"{'Observable':<{col_w}}  {'t0 (idx)':>8}  {'t0 (time)':>12}  {'Mean':>16}  "
            f"{'SD':>12}  {'SEM':>12}  {'tau_int':>10}  {'Neff':>8}",
            "-" * (col_w + 91),
        ]
        for r in results:
            rows.append(
                f"{r['label']:<{col_w}}  {r['t0']:>8d}  {r['t0_time']:>12.4f}  {r['mean']:>16.6f}  "
                f"{r['sd']:>12.6f}  {r['SEM']:>12.6f}  {r['tau_int']:>10.2f}  {r['Neff']:>8.1f}"
            )
    combined = header_text + "\n".join(rows) + "\n"
    print(combined)
    (output_dir / "summary_combined.txt").write_text(combined)
    try:
        parent_dir = log_path.resolve().parent
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
