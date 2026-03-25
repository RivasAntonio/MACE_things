#!/usr/bin/env python3
"""
calculate_msd.py
----------------
Calculate the Mean Square Displacement (MSD) with respect to the first frame
of a trajectory file compatible with ASE.

Usage:
    python calculate_msd.py trajectory.xyz [options]

MSD(t) = (1/N) * sum_i |r_i(t) - r_i(0)|^2
"""

import argparse
import numpy as np
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate MSD with respect to the first frame of a trajectory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("trajectory", help="Trajectory file (any ASE-readable format)")
    parser.add_argument(
        "--index",
        default=":",
        help="ASE index string to select frames (e.g. '0:100', '::2', ':')",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        default=None,
        help="Chemical symbols to include (e.g. --species Si O). Default: all atoms.",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=1.0,
        help="Time step between frames (in fs). Used only for the x-axis label.",
    )
    parser.add_argument(
        "--no-pbc",
        action="store_true",
        help="Disable periodic boundary conditions correction (minimum image convention).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file for MSD data (CSV). Default: <trajectory_stem>_msd.csv",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not show the MSD plot.",
    )
    parser.add_argument(
        "--save-plot",
        nargs="?",
        const="",
        default=None,
        metavar="FILE",
        help="Save plot to FILE (e.g. msd.png). If given without a filename, uses <trajectory>.png.",
    )
    parser.add_argument(
        "--format",
        default=None,
        help="ASE format string (e.g. 'vasp', 'lammps-dump-text'). Auto-detected if not given.",
    )
    return parser.parse_args()


def minimum_image(dr, cell):
    """Apply minimum image convention to displacement vectors.
    
    Parameters
    ----------
    dr : np.ndarray, shape (N, 3)
        Displacement vectors in Cartesian coordinates.
    cell : np.ndarray, shape (3, 3)
        Unit cell matrix (rows are lattice vectors).
    
    Returns
    -------
    np.ndarray, shape (N, 3)
        Corrected displacement vectors.
    """
    # Convert to fractional coordinates
    inv_cell = np.linalg.inv(cell)
    dr_frac = dr @ inv_cell
    # Wrap to [-0.5, 0.5)
    dr_frac -= np.round(dr_frac)
    # Convert back to Cartesian
    return dr_frac @ cell


def compute_msd(frames, species=None, use_pbc=True):
    """Compute per-species and total MSD relative to the first frame.

    Parameters
    ----------
    frames : list of ase.Atoms
        List of ASE Atoms objects.
    species : list of str or None
        Chemical symbols to include. None means all.
    use_pbc : bool
        Whether to apply minimum image convention.

    Returns
    -------
    times : np.ndarray
        Frame indices (0, 1, 2, ...).
    msd_total : np.ndarray
        Total MSD over selected atoms at each frame.
    msd_by_species : dict
        {symbol: np.ndarray} MSD for each species.
    """
    ref = frames[0]
    all_symbols = np.array(ref.get_chemical_symbols())

    if species is None:
        mask = np.ones(len(ref), dtype=bool)
        selected_species = sorted(set(all_symbols))
    else:
        mask = np.isin(all_symbols, species)
        selected_species = [s for s in species if s in all_symbols]
        unknown = [s for s in species if s not in all_symbols]
        if unknown:
            print(f"Warning: species {unknown} not found in trajectory.", file=sys.stderr)

    if not np.any(mask):
        sys.exit("Error: no atoms match the requested species.")

    ref_pos = ref.get_positions()[mask]
    cell = ref.get_cell()
    has_cell = cell.volume > 0

    n_frames = len(frames)
    msd_total = np.zeros(n_frames)
    msd_by_species = {s: np.zeros(n_frames) for s in selected_species}
    counts = {s: np.sum(all_symbols[mask] == s) for s in selected_species}

    for i, atoms in enumerate(frames):
        pos = atoms.get_positions()[mask]
        dr = pos - ref_pos

        if use_pbc and has_cell:
            dr = minimum_image(dr, np.array(cell))

        sq_disp = np.sum(dr**2, axis=1)  # shape (N,)
        msd_total[i] = np.mean(sq_disp)

        sym_sel = all_symbols[mask]
        for s in selected_species:
            idx = sym_sel == s
            if counts[s] > 0:
                msd_by_species[s][i] = np.mean(sq_disp[idx])

    return np.arange(n_frames), msd_total, msd_by_species


def main():
    args = parse_args()

    try:
        from ase.io import read
    except ImportError:
        sys.exit("ASE is required. Install it with: pip install ase")

    traj_path = Path(args.trajectory)
    if not traj_path.exists():
        sys.exit(f"Error: file not found: {traj_path}")

    print(f"Reading trajectory: {traj_path}")
    fmt_kwargs = {"format": args.format} if args.format else {}
    frames = read(str(traj_path), index=args.index, **fmt_kwargs)
    if not isinstance(frames, list):
        frames = [frames]
    print(f"  {len(frames)} frames loaded.")

    use_pbc = not args.no_pbc
    frame_idx, msd_total, msd_by_species = compute_msd(
        frames, species=args.species, use_pbc=use_pbc
    )

    times = frame_idx * args.timestep  # in fs

    # ---------- Print summary ----------
    print(f"\nMSD summary (first frame = 0):")
    print(f"  {'Frame':>6}  {'Time (fs)':>12}  {'MSD_total (Å²)':>18}")
    step = max(1, len(frames) // 10)
    for i in range(0, len(frames), step):
        print(f"  {frame_idx[i]:>6}  {times[i]:>12.2f}  {msd_total[i]:>18.4f}")

    # ---------- Save CSV ----------
    if args.output:
        output_path = args.output
        header = "frame,time_fs,MSD_total_A2"
        cols = [frame_idx, times, msd_total]
        col_fmts = ["%d", "%.4f", "%.6f"]
        for s, msd_s in msd_by_species.items():
            header += f",MSD_{s}_A2"
            cols.append(msd_s)
            col_fmts.append("%.6f")

        data = np.column_stack(cols)
        np.savetxt(output_path, data, delimiter=",", header=header, comments="", fmt=col_fmts)
        print(f"\nMSD data saved to: {output_path}")

    # ---------- Plot ----------
    if (not args.no_plot) or args.save_plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available. Skipping plot.", file=sys.stderr)
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(times, msd_total, "k-", lw=2, label="Total")
        for s, msd_s in msd_by_species.items():
            ax.plot(times, msd_s, lw=1.5, label=s)

        ax.set_xlabel("Time (fs)" if args.timestep != 1.0 else "Frame")
        ax.set_ylabel("MSD (Å²)")
        ax.set_title(f"MSD — {traj_path.name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if args.save_plot is not None:
            save_path = args.save_plot or (traj_path.stem + "_msd.png")
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to: {save_path}")
        elif not args.no_plot:
            plt.show()


if __name__ == "__main__":
    main()
