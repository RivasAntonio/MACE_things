#!/usr/bin/env python3
"""
ase-rdf.py  –  Radial Distribution Function calculator using ASE
Computes g(r) and the coordination number N(r) for one or more element pairs.
"""

import sys
import argparse
import os
import itertools

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase.geometry.analysis import Analysis


# ── Helpers ──────────────────────────────────────────────────────────────────

def log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg)


def available_pairs(elements: list[str]) -> list[tuple[str, str]]:
    """Return all unique sorted element pairs (including same-element pairs)."""
    return [tuple(sorted(p)) for p in itertools.combinations_with_replacement(elements, 2)]


def select_elements_interactive(elements: list[str]) -> tuple[str, str]:
    """Prompt the user to choose two elements."""
    print(f"\n🔹 Available elements: {', '.join(elements)}")
    opts = "/".join(elements)
    for label in ("First", "Second"):
        while True:
            e = input(f"{label} element ({opts}): ").strip()
            if e in elements:
                break
            print(f"  ❌ '{e}' not valid. Choose from: {', '.join(elements)}")
        if label == "First":
            e1 = e
        else:
            e2 = e
    return tuple(sorted([e1, e2]))


def parse_rdf_result(
    result, rmax: float, nbins: int, quiet: bool
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalise whatever get_rdf() returns into (r, g_r) 1-D arrays.
    Handles: 2-tuple, 1-D array, and 2-D array (nframes × nbins or nbins × nframes).
    """
    if isinstance(result, tuple) and len(result) >= 2:
        r = np.asarray(result[0]).ravel()
        g_r = np.asarray(result[1]).ravel()
        log(f"   RDF returned as tuple  ({len(r)} bins)", quiet)
        return r, g_r

    raw = np.asarray(result)

    if raw.ndim == 1:
        g_r = raw
        r = np.linspace(0.0, rmax, len(g_r))
        log(f"   RDF returned as 1-D array  ({len(r)} bins)", quiet)
        return r, g_r

    if raw.ndim == 2:
        # Determine which axis is the bin axis
        if raw.shape[0] == nbins:          # shape (nbins, nframes)
            g_r = raw.mean(axis=1)
            nframes = raw.shape[1]
        else:                              # shape (nframes, nbins)
            g_r = raw.mean(axis=0)
            nframes = raw.shape[0]
        r = np.linspace(0.0, rmax, len(g_r))
        log(f"   RDF returned as 2-D array  ({len(r)} bins, averaged over {nframes} frames)", quiet)
        return r, g_r

    raise ValueError(f"Unexpected RDF result shape: {raw.shape}")


def get_rdf(
    analysis: Analysis,
    rmax: float,
    nbins: int,
    e1: str,
    e2: str,
    quiet: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute g(r) with a graceful fallback if element-specific call fails."""
    try:
        result = analysis.get_rdf(rmax=rmax, nbins=nbins, elements=(e1, e2))
        return parse_rdf_result(result, rmax, nbins, quiet)
    except Exception as err:
        log(f"   ⚠️  Element-specific RDF failed ({err}); trying without element filter…", quiet)

    result = analysis.get_rdf(rmax=rmax, nbins=nbins)
    return parse_rdf_result(result, rmax, nbins, quiet)


def adaptive_rmax(traj, safety: float = 0.5) -> float:
    """
    Return the largest physically valid rmax for a periodic cell.

    For any Bravais lattice the minimum-image convention requires r < L_perp/2,
    where L_perp is the perpendicular width of the cell along each axis:

        L_perp_i = V / |a_j × a_k|          (volume / face area)

    Using lattice-vector lengths instead would over-estimate rmax for
    non-orthogonal (triclinic) cells.

    Parameters
    ----------
    safety : float
        Multiplicative factor applied to the raw limit (default 0.5 = exact limit).
        Use e.g. 0.45 for a small margin away from the hard boundary.
    """
    atoms = traj[0]
    cell = atoms.cell.array          # 3×3, rows are lattice vectors
    volume = atoms.get_volume()

    perp_widths = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        face_normal_area = np.linalg.norm(np.cross(cell[j], cell[k]))
        if face_normal_area > 0:
            perp_widths.append(volume / face_normal_area)

    return min(perp_widths) * safety


def coordination_number(
    r: np.ndarray,
    g_r: np.ndarray,
    traj,
    e2: str,
) -> np.ndarray:
    """
    Cumulative coordination number:

        N(r) = 4π ρ_B ∫₀ʳ g(r') r'² dr'

    ρ_B is the number density of the *target* species (e2), not all atoms.
    This gives the mean number of e2 neighbours within radius r of an e1 atom.
    """
    atoms = traj[0]
    symbols = atoms.get_chemical_symbols()
    n_B = symbols.count(e2)
    volume = atoms.get_volume()
    rho_B = n_B / volume                    # density of species B (Å⁻³)

    dr = r[1] - r[0] if len(r) > 1 else 1.0
    N_r = 4.0 * np.pi * rho_B * np.cumsum(g_r * r**2 * dr)
    return N_r


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def plot_pairs(
    pairs_data: list[dict],
    outdir: str,
    traj_basename: str,
    no_show: bool,
    quiet: bool,
) -> None:
    """
    Create one combined figure (g(r) left, N(r) right) for all pairs,
    plus individual PNGs per pair.
    """
    n = len(pairs_data)

    # ── Combined figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_g, ax_n = axes

    for i, pd in enumerate(pairs_data):
        c = COLORS[i % len(COLORS)]
        label = f"{pd['e1']}-{pd['e2']}"
        ax_g.plot(pd["r"], pd["g_r"], color=c, lw=2, label=label)
        ax_n.plot(pd["r"], pd["N_r"], color=c, lw=2, label=label)

    for ax, ylabel, title in [
        (ax_g, "g(r)", "Radial Distribution Function"),
        (ax_n, "N(r)", "Cumulative Coordination Number"),
    ]:
        ax.set_xlabel("r  [Å]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle(traj_basename, fontsize=11, y=1.01)
    plt.tight_layout()

    pairs_tag = "_".join(f"{pd['e1']}{pd['e2']}" for pd in pairs_data)
    combined_png = os.path.join(outdir, f"rdf_{traj_basename}_{pairs_tag}.png")
    plt.savefig(combined_png, dpi=300, bbox_inches="tight")
    log(f"   🖼️  Combined plot saved → {combined_png}", quiet)

    if not no_show:
        plt.show()
    else:
        plt.close()

    # ── Individual figures ───────────────────────────────────────────────────
    if n > 1:
        for pd in pairs_data:
            fig2, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
            label = f"{pd['e1']}-{pd['e2']}"
            a1.plot(pd["r"], pd["g_r"], lw=2, label=label)
            a2.plot(pd["r"], pd["N_r"], "r-", lw=2, label=f"N(r): {label}")
            for ax, ylabel, title in [
                (a1, "g(r)", f"RDF: {label}"),
                (a2, "N(r)", f"Coordination number: {label}"),
            ]:
                ax.set_xlabel("r  [Å]")
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.legend()
            plt.tight_layout()
            indiv_png = os.path.join(outdir, f"rdf_{traj_basename}_{pd['e1']}_{pd['e2']}.png")
            plt.savefig(indiv_png, dpi=300, bbox_inches="tight")
            plt.close()
            log(f"   🖼️  Individual plot saved → {indiv_png}", quiet)


# ── Main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute the Radial Distribution Function (RDF) and coordination number N(r).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python ase-rdf.py traj.traj                      # interactive element selection
  python ase-rdf.py traj.traj Si O                 # one specific pair
  python ase-rdf.py traj.traj Si O Si Si           # two pairs at once
  python ase-rdf.py traj.traj --all-pairs          # every unique pair
  python ase-rdf.py traj.traj Si O --rmax 8 --nbins 200 --no-show
  python ase-rdf.py traj.traj Si O --quiet --csv""",
    )
    p.add_argument("traj_file", help="Trajectory file (.traj, .xyz, …)")
    p.add_argument(
        "elements",
        nargs="*",
        help="Element symbols (0, 2, or any even number for multiple pairs)",
    )
    p.add_argument("--all-pairs", action="store_true", help="Compute all unique element pairs")
    p.add_argument("--rmax", type=float, default=None,
                   help="Maximum radius in Å. Defaults to half the smallest cell "
                        "perpendicular width (auto-detected from the trajectory).")
    p.add_argument("--safety", type=float, default=0.5,
                   help="Fraction of the cell perpendicular width used as rmax "
                        "when --rmax is not set (default: 0.5 = exact MIC limit). "
                        "Use e.g. 0.45 for a small safety margin.")
    p.add_argument("--nbins", type=int, default=100, help="Number of histogram bins (default: 100)")
    p.add_argument("--outdir", default="rdf_results", help="Output directory (default: rdf_results)")
    p.add_argument("--no-show", action="store_true", help="Save plots without displaying them")
    p.add_argument("--csv", action="store_true", help="Also save data as CSV (default: .dat)")
    p.add_argument("--quiet", "-q", action="store_true", help="Suppress informational output")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not os.path.exists(args.traj_file):
        sys.exit(f"❌  File not found: {args.traj_file}")

    if args.elements and len(args.elements) % 2 != 0:
        sys.exit("❌  Provide an even number of element symbols (pairs), e.g. Si O Si Si")

    # ── Load trajectory ───────────────────────────────────────────────────────
    log(f"📂  Reading trajectory: {args.traj_file}", args.quiet)
    traj = read(args.traj_file, ":")
    log(f"    {len(traj)} frame(s) loaded", args.quiet)

    elements = sorted(set(traj[0].get_chemical_symbols()))
    log(f"🔹  Elements in trajectory: {', '.join(elements)}", args.quiet)

    traj_basename = os.path.splitext(os.path.basename(args.traj_file))[0]
    os.makedirs(args.outdir, exist_ok=True)

    # ── Resolve rmax ──────────────────────────────────────────────────────────
    if args.rmax is not None:
        rmax = args.rmax
        log(f"📏  rmax = {rmax:.3f} Å  (user-specified)", args.quiet)
    else:
        rmax = adaptive_rmax(traj, safety=args.safety)
        log(f"📏  rmax = {rmax:.3f} Å  (auto: {args.safety} × smallest cell perpendicular width)",
            args.quiet)

    # ── Determine pairs to compute ────────────────────────────────────────────
    if args.all_pairs:
        pairs = available_pairs(elements)
        log(f"📐  Computing all {len(pairs)} unique pair(s): {pairs}", args.quiet)

    elif len(args.elements) >= 2:
        pairs = []
        it = iter(args.elements)
        for e1, e2 in zip(it, it):
            for e in (e1, e2):
                if e not in elements:
                    sys.exit(f"❌  Element '{e}' not found in trajectory. "
                             f"Available: {', '.join(elements)}")
            pairs.append(tuple(sorted([e1, e2])))
        log(f"📐  Pair(s) requested: {pairs}", args.quiet)

    else:
        if args.quiet:
            sys.exit("❌  Quiet mode requires elements to be specified explicitly.")
        pair = select_elements_interactive(elements)
        pairs = [pair]
        log(f"✅  Selected pair: {pair[0]}-{pair[1]}", args.quiet)

    # ── Compute RDF for each pair ─────────────────────────────────────────────
    log(f"\n⚙️   rmax={rmax:.3f} Å   nbins={args.nbins}", args.quiet)
    analysis = Analysis(traj)
    pairs_data = []

    for e1, e2 in pairs:
        log(f"\n🔬  Computing RDF  {e1}-{e2} …", args.quiet)
        try:
            r, g_r = get_rdf(analysis, rmax, args.nbins, e1, e2, args.quiet)
        except Exception as err:
            print(f"   ❌  Could not compute RDF for {e1}-{e2}: {err}")
            continue

        N_r = coordination_number(r, g_r, traj, e2)
        log(f"   ✅  {len(r)} points, r ∈ [{r[0]:.3f}, {r[-1]:.3f}] Å", args.quiet)

        # Save data file
        ext = "csv" if args.csv else "dat"
        delimiter = "," if args.csv else "   "
        data_file = os.path.join(args.outdir, f"rdf_{traj_basename}_{e1}_{e2}.{ext}")
        header = (
            f"RDF  {e1}-{e2}  |  trajectory: {args.traj_file}  "
            f"|  rmax={rmax:.3f} Å  nbins={args.nbins}\n"
            + ("r_Angstrom,g_r,N_r" if args.csv else "r(Å)          g(r)          N(r)")
        )
        np.savetxt(
            data_file,
            np.column_stack((r, g_r, N_r)),
            delimiter=delimiter,
            header=header,
            fmt="%.6f",
        )
        log(f"   💾  Data saved → {data_file}", args.quiet)

        pairs_data.append({"e1": e1, "e2": e2, "r": r, "g_r": g_r, "N_r": N_r})

    if not pairs_data:
        sys.exit("❌  No RDF data was successfully computed.")

    # ── Plot ──────────────────────────────────────────────────────────────────
    log(f"\n📊  Generating plot(s) …", args.quiet)
    plot_pairs(pairs_data, args.outdir, traj_basename, args.no_show, args.quiet)

    log(f"\n✅  All results saved in '{args.outdir}/'", args.quiet)


if __name__ == "__main__":
    main()