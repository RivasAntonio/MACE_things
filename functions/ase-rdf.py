#!/usr/bin/env python3
"""
ase-rdf.py - RDF + coordination number using ASE (modern API)
"""

import argparse
import itertools
import os

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.geometry.rdf import get_rdf

plt.rcParams.update({'font.size': 16})


# -- Helpers ------------------------------------------------------------------

def log(msg, quiet=False):
    if not quiet:
        print(msg)


def available_pairs(elements):
    return [tuple(sorted(p)) for p in itertools.combinations_with_replacement(elements, 2)]


def select_elements_interactive(elements):
    print(f"\nAvailable elements: {', '.join(elements)}")
    opts = "/".join(elements)

    e1 = e2 = None
    for label in ("First", "Second"):
        while True:
            e = input(f"{label} element ({opts}): ").strip()
            if e in elements:
                break
            print(f"'{e}' is not a valid element symbol.")
        if label == "First":
            e1 = e
        else:
            e2 = e

    return tuple(sorted([e1, e2]))


def compute_rdf(traj, rmax, nbins, e1, e2, quiet):
    """Average RDF over all frames in the trajectory."""
    g_accum = None
    count = 0

    for atoms in traj:
        try:
            g, r = get_rdf(atoms, rmax=rmax, nbins=nbins, elements=(e1, e2))
        except Exception as err:
            log(f"Frame skipped: {err}", quiet)
            continue

        if g_accum is None:
            g_accum = np.zeros_like(g)

        g_accum += g
        count += 1

    if count == 0:
        raise RuntimeError("No valid frames found for RDF calculation.")

    return r, g_accum / count


def _cell_min_perpendicular_width(atoms):
    """
    Return the minimum perpendicular width of the simulation cell.

    For a triclinic cell with lattice vectors a, b, c the perpendicular
    width along direction i is:

        h_i = V / |b_j x b_k|

    where V is the cell volume and j, k are the two complementary indices.
    This is the largest sphere diameter that fits inside the cell along
    each axis and therefore sets the upper bound on rmax via the
    minimum-image convention.
    """
    cell = atoms.cell.array
    volume = atoms.get_volume()

    widths = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        area = np.linalg.norm(np.cross(cell[j], cell[k]))
        widths.append(volume / area)

    return min(widths)


def adaptive_rmax(traj, quiet=False):
    """
    Compute the maximum physically valid rmax for the entire trajectory.

    The hard constraint imposed by the minimum-image convention is:

        rmax < h_min / 2

    where h_min is the minimum perpendicular cell width.  For trajectories
    run under constant-pressure (NPT) conditions the cell fluctuates, so
    the constraint must be evaluated on every frame and the global minimum
    must be used.  Using only the first frame would produce an rmax that
    violates the constraint for frames with smaller cells.

    The factor 0.5 is exact - it is not a safety margin.
    """
    min_width = np.inf

    for atoms in traj:
        w = _cell_min_perpendicular_width(atoms)
        if w < min_width:
            min_width = w

    rmax = min_width * 0.5
    log(f"Adaptive rmax = {rmax:.4f} A  "
        f"(half of minimum perpendicular cell width {min_width:.4f} A "
        f"over {len(traj)} frames)", quiet)
    return rmax


def coordination_number(r, g, traj, e2):
    """
    Running coordination number N(r) = 4 pi rho integral_0^r g(r') r'^2 dr'

    The number density rho is taken from the first frame.  For NPT
    trajectories an average density would be more rigorous; adjust if needed.
    """
    atoms = traj[0]
    symbols = atoms.get_chemical_symbols()
    rho = symbols.count(e2) / atoms.get_volume()
    dr = r[1] - r[0]
    return 4 * np.pi * rho * np.cumsum(g * r**2 * dr)


def first_two_peaks(r, g):
    """Return the r positions of the first two local maxima in g(r)."""
    peak_indices = []

    for i in range(1, len(g) - 1):
        if g[i] > g[i - 1] and g[i] >= g[i + 1]:
            peak_indices.append(i)

    if not peak_indices:
        return []

    peak_indices = sorted(peak_indices, key=lambda idx: g[idx], reverse=True)
    peak_indices = sorted(peak_indices[:2])
    return [r[idx] for idx in peak_indices]


# -- Plot ---------------------------------------------------------------------

def plot_pairs(data, outdir, name, quiet, show):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for d in data:
        label = f"{d['e1']}-{d['e2']}"
        ax1.plot(d["r"], d["g"], label=label)
        ax2.plot(d["r"], d["N"], label=label)

    ax1.set_xlabel("r (A)", fontsize=20)
    ax1.set_ylabel("g(r)", fontsize=20)
    ax2.set_xlabel("r (A)", fontsize=20)
    ax2.set_ylabel("N(r)", fontsize=20)

    ax1.legend(fontsize=16)
    ax2.legend(fontsize=16)

    plt.tight_layout()
    outfile = os.path.join(outdir, f"rdf_{name}.png")
    plt.savefig(outfile, dpi=300)

    if show:
        plt.show()

    log(f"Saved plot -> {outfile}", quiet)


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute pair RDF and coordination number from an ASE-readable trajectory."
    )
    parser.add_argument("traj_file", help="Trajectory file (any ASE-readable format).")
    parser.add_argument(
        "elements", nargs="*",
        help="Element pairs to compute, e.g. 'Li O Na O'. "
             "If omitted, an interactive prompt is shown."
    )
    parser.add_argument("--nbins", type=int, default=50, help="Number of RDF bins.")
    parser.add_argument("--stride", type=int, default=1, help="Read every N-th frame.")
    parser.add_argument(
        "--rmax", type=float, default=None,
        help="Override rmax (Angstrom). "
             "Default: maximum allowed by the minimum-image convention "
             "evaluated over all frames."
    )
    parser.add_argument("--outdir", default="rdf_results", help="Output directory.")
    parser.add_argument("--no-show", dest="show", action="store_false")
    parser.add_argument("--quiet", action="store_true")
    parser.set_defaults(show=True)

    args = parser.parse_args()

    if args.stride < 1:
        parser.error("--stride must be >= 1")

    traj = read(args.traj_file, f"::{args.stride}")
    log(f"Loaded {len(traj)} frames from '{args.traj_file}' (stride={args.stride}).",
        args.quiet)

    elements = sorted(set(traj[0].get_chemical_symbols()))

    if args.elements:
        pairs = []
        it = iter(args.elements)
        for e1, e2 in zip(it, it):
            pairs.append((e1, e2))
    else:
        pairs = [select_elements_interactive(elements)]

    if args.rmax is not None:
        rmax = args.rmax
        log(f"Using user-supplied rmax = {rmax:.4f} A.", args.quiet)
    else:
        rmax = adaptive_rmax(traj, args.quiet)

    os.makedirs(args.outdir, exist_ok=True)

    results = []
    for e1, e2 in pairs:
        log(f"Computing RDF for pair {e1}-{e2} ...", args.quiet)

        r, g = compute_rdf(traj, rmax, args.nbins, e1, e2, args.quiet)
        peaks = first_two_peaks(r, g)
        if peaks:
            print(
                f"First peak positions for {e1}-{e2}: "
                + ", ".join(f"{peak:.4f} A" for peak in peaks)
            )
        else:
            print(f"No local maxima found for {e1}-{e2}.")
        N = coordination_number(r, g, traj, e2)

        outfile = os.path.join(args.outdir, f"rdf_{e1}_{e2}.dat")
        np.savetxt(outfile, np.column_stack((r, g, N)),
                   header="r(A)  g(r)  N(r)")
        log(f"Data saved -> {outfile}", args.quiet)

        results.append({"e1": e1, "e2": e2, "r": r, "g": g, "N": N})

    plot_pairs(results, args.outdir, os.path.basename(args.traj_file),
               args.quiet, args.show)


if __name__ == "__main__":
    main()
