#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SiO2 bond counter and structure comparator using ASE.

Counts all bonds within a given cutoff (default 2.0 A) and checks
whether two structures share the same bond topology.

Usage
-----
    python sio2_bond_checker.py structure1.cif structure2.cif [--cutoff 2.0]

Supported formats: anything ASE can read (cif, poscar, extxyz, etc.)
"""

import argparse
import sys
from collections import Counter

import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def count_bonds(atoms, cutoff: float = 2.0) -> dict:
    """
    Count all unique bonds within *cutoff* angstroms using ASE's
    neighbor_list (scipy KD-tree under the hood, O(N log N)).

    Parameters
    ----------
    atoms   : ase.Atoms
    cutoff  : float, bond distance threshold in angstroms

    Returns
    -------
    dict with keys:
        "total"          : int   – total number of unique bonds
        "by_pair"        : Counter{(sym_A, sym_B): count}
        "distances_mean" : dict{(sym_A, sym_B): mean_distance}
        "distances_std"  : dict{(sym_A, sym_B): std_distance}
    """
    symbols = np.array(atoms.get_chemical_symbols())

    # neighbor_list returns indices i, j and distances d for ALL pairs
    # with d < cutoff.  Each pair appears twice (i->j and j->i), so we
    # keep only i < j to avoid double-counting.
    i_idx, j_idx, distances = neighbor_list("ijd", atoms, cutoff)

    mask = i_idx < j_idx
    i_idx    = i_idx[mask]
    j_idx    = j_idx[mask]
    distances = distances[mask]

    # Build per-pair-type accumulators
    pair_labels = [
        tuple(sorted([symbols[i], symbols[j]]))
        for i, j in zip(i_idx, j_idx)
    ]

    pair_counter = Counter(pair_labels)

    dist_by_pair: dict[tuple, list] = {}
    for label, d in zip(pair_labels, distances):
        dist_by_pair.setdefault(label, []).append(d)

    return {
        "total": int(mask.sum()),
        "by_pair": pair_counter,
        "distances_mean": {k: float(np.mean(v)) for k, v in dist_by_pair.items()},
        "distances_std":  {k: float(np.std(v))  for k, v in dist_by_pair.items()},
    }


def compare_bond_counts(info1: dict, info2: dict) -> dict:
    """
    Compare bond-count dictionaries from two structures.

    Returns a result dict with a boolean 'match' field and a human-readable
    'report' string.
    """
    lines = []
    match = True

    all_pairs = set(info1["by_pair"]) | set(info2["by_pair"])

    for pair in sorted(all_pairs):
        n1 = info1["by_pair"].get(pair, 0)
        n2 = info2["by_pair"].get(pair, 0)
        status = "OK" if n1 == n2 else "MISMATCH"
        if n1 != n2:
            match = False
        lines.append(f"  {'-'.join(pair):<8}  struct1={n1:>6d}  struct2={n2:>6d}  [{status}]")

    if info1["total"] != info2["total"]:
        match = False

    lines.append(f"\n  Total      struct1={info1['total']:>6d}  struct2={info2['total']:>6d}  "
                 f"[{'OK' if match else 'MISMATCH'}]")

    return {"match": match, "report": "\n".join(lines)}


def print_bond_info(label: str, info: dict, cutoff: float) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}  (cutoff = {cutoff} A)")
    print(f"{'='*60}")
    print(f"  {'Pair':<8}  {'Count':>8}  {'Mean d (A)':>12}  {'Std d (A)':>12}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*12}")
    for pair in sorted(info["by_pair"]):
        label_str = "-".join(pair)
        n    = info["by_pair"][pair]
        mean = info["distances_mean"][pair]
        std  = info["distances_std"][pair]
        print(f"  {label_str:<8}  {n:>8d}  {mean:>12.4f}  {std:>12.4f}")
    print(f"\n  Total unique bonds: {info['total']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Count SiO2 bonds and compare two structures."
    )
    p.add_argument("structure1", help="Reference SiO2 structure file")
    p.add_argument("structure2", help="Second structure to compare")
    p.add_argument(
        "--cutoff", type=float, default=2.0,
        help="Bond distance cutoff in angstroms (default: 2.0)"
    )
    p.add_argument(
        "--index", type=int, default=0,
        help="Frame index for multi-frame files (default: 0)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"\nReading '{args.structure1}' ...")
    atoms1 = read(args.structure1, index=args.index)
    print(f"  {len(atoms1)} atoms  |  formula: {atoms1.get_chemical_formula()}")

    print(f"Reading '{args.structure2}' ...")
    atoms2 = read(args.structure2, index=args.index)
    print(f"  {len(atoms2)} atoms  |  formula: {atoms2.get_chemical_formula()}")

    print(f"\nComputing bonds (cutoff = {args.cutoff} A) ...")
    info1 = count_bonds(atoms1, cutoff=args.cutoff)
    info2 = count_bonds(atoms2, cutoff=args.cutoff)

    print_bond_info(f"Structure 1: {args.structure1}", info1, args.cutoff)
    print_bond_info(f"Structure 2: {args.structure2}", info2, args.cutoff)

    result = compare_bond_counts(info1, info2)
    print(f"\n{'='*60}")
    print("  COMPARISON")
    print(f"{'='*60}")
    print(result["report"])
    print(f"\n  Result: {'STRUCTURES HAVE THE SAME BOND COUNT' if result['match'] else 'BOND COUNTS DIFFER'}")

    sys.exit(0 if result["match"] else 1)


if __name__ == "__main__":
    main()
