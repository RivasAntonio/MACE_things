#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified script for generating datasets from VASP files with different sampling strategies.
Combines the functionality of generate_first_last_files.py, generate_interval_files.py, and generate_train_files.py
Integrated with collect.py functionality for proper energy extraction and PSTRESS correction.
"""
import argparse
import os
import random
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from ase.io import write
from ase.io import read as ase_read
from pymatgen.io.vasp.outputs import Vasprun

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc=None, total=None, disable=False):
        return iterable


# Constants
KBAR_TO_EV_A3 = 1.0 / 1602.1766208  # 1 eV/Å^3 = 1602.1766208 kbar


def pick_energy(step: dict, prefer=("e_0_energy", "e_fr_energy")) -> Tuple[Optional[float], Optional[str]]:
    """Pick energy from ionic step in order of preference."""
    for k in prefer:
        if k in step and step[k] is not None:
            return float(step[k]), k
    return None, None


def ase_pmg_same_frame(ase_atoms, pmg_structure) -> bool:
    """Check if ASE and pymatgen represent the same frame (cell + cartesian coords)."""
    cell_ase = np.asarray(ase_atoms.cell.array, dtype=float)
    cell_pmg = np.asarray(pmg_structure.lattice.matrix, dtype=float)
    if not np.allclose(cell_ase, cell_pmg, rtol=1e-8, atol=1e-6):
        return False

    pos_ase = np.asarray(ase_atoms.get_positions(), dtype=float)
    pos_pmg = np.asarray(pmg_structure.cart_coords, dtype=float)
    if pos_ase.shape != pos_pmg.shape:
        return False
    if not np.allclose(pos_ase, pos_pmg, rtol=1e-8, atol=1e-5):
        return False

    sym_ase = list(ase_atoms.get_chemical_symbols())
    sym_pmg = [str(sp) for sp in pmg_structure.species]
    return sym_ase == sym_pmg


def load_atoms_from_file(filename: str, fmt: str = None, use_enhanced: bool = False) -> List:
    """
    Load a list of Atoms from a file.
    If use_enhanced=True and the file is a vasprun.xml, uses enhanced processing with pymatgen.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")

    if use_enhanced and filename.endswith("vasprun.xml"):
        return load_vasprun_enhanced(filename)

    if fmt:
        atoms_list = ase_read(filename, index=":", format=fmt)
    else:
        try:
            atoms_list = ase_read(filename, index=":")
        except Exception as e:
            raise ValueError(f"Cannot automatically guess the file format: {e}")

    for i, atoms in enumerate(atoms_list):
        atoms.info["V_A3"] = float(atoms.get_volume())
        atoms.info["ionic_step"] = i + 1
    return atoms_list


def load_vasprun_enhanced(filename: str, prefer_energy=("e_0_energy", "e_fr_energy"),
                          verbose: bool = False) -> List:
    """
    Enhanced processing of vasprun.xml using both pymatgen (for energy) and ASE (for forces/stress).
    Applies PSTRESS correction when ISIF=3 and IBRION=0.
    Returns list of Atoms with REF_energy, REF_forces, REF_stress, REF_virial.
    """
    xml_path = Path(filename)

    vr = Vasprun(
        str(xml_path),
        parse_dos=False,
        parse_eigen=False,
        parse_projected_eigen=False,
        parse_potcar_file=False,
        exception_on_bad_xml=False,
    )
    pmg_steps = vr.ionic_steps
    n_pmg = len(pmg_steps)

    ase_frames = ase_read(str(xml_path), index=":", format="vasp-xml")
    n_ase = len(ase_frames)

    n = min(n_pmg, n_ase)
    if verbose:
        print(f"  [Enhanced] steps: pymatgen={n_pmg}  ase={n_ase}  using min={n}")

    incar = vr.incar or {}
    isif = int(incar.get("ISIF", 0)) if "ISIF" in incar else 0
    ibrion = int(incar.get("IBRION", 0)) if "IBRION" in incar else 0
    apply_pv = (isif == 3 and ibrion == 0)
    pstress_kbar = float(incar.get("PSTRESS", 0.0)) if "PSTRESS" in incar else 0.0

    if verbose and apply_pv:
        print(f"  [Enhanced] ISIF={isif}, IBRION={ibrion}, PSTRESS={pstress_kbar} kbar -> Applying -pV correction")

    out_frames = []
    skipped = 0

    for i in range(n):
        step = pmg_steps[i]
        ase_atoms = ase_frames[i]

        pmg_structure = step.get("structure", None)
        if pmg_structure is None:
            skipped += 1
            continue

        E, Ekey = pick_energy(step, prefer=prefer_energy)
        if E is None:
            skipped += 1
            continue

        if not ase_pmg_same_frame(ase_atoms, pmg_structure):
            skipped += 1
            if verbose:
                print(f"  [Enhanced] Frame {i} mismatch between ASE and pymatgen, skipping")
            continue

        try:
            F = np.asarray(ase_atoms.get_forces(apply_constraint=False), dtype=float)
        except Exception:
            skipped += 1
            continue

        try:
            S = np.asarray(ase_atoms.get_stress(voigt=False), dtype=float)
            if S.shape != (3, 3):
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        if (not np.isfinite(E)) or (not np.all(np.isfinite(F))) or (not np.all(np.isfinite(S))):
            skipped += 1
            continue

        V = float(ase_atoms.get_volume())
        pV_pstress_eV = pstress_kbar * V * KBAR_TO_EV_A3
        virial = -S * V  # eV

        ase_atoms.info["REF_energy_key_used"] = Ekey
        ase_atoms.arrays["REF_forces"] = F
        ase_atoms.info["REF_stress"] = S.reshape(9)
        ase_atoms.info["REF_virial"] = virial.reshape(9)
        ase_atoms.info["V_A3"] = V

        if apply_pv:
            ase_atoms.info["REF_energy"] = float(E - pV_pstress_eV)
            ase_atoms.info["PSTRESS_kbar"] = float(pstress_kbar)
            ase_atoms.info["PSTRESS_pV_eV"] = float(pV_pstress_eV)
        else:
            ase_atoms.info["REF_energy"] = float(E)

        ase_atoms.calc = None
        out_frames.append(ase_atoms)

    if verbose and skipped > 0:
        print(f"  [Enhanced] Skipped {skipped} frames due to missing data or mismatches")

    return out_frames


def find_vasp_files(data_path, file_type):
    """Find VASP files of specified type in the data path."""
    vasp_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file_type == "xml" and file.endswith("run.xml"):
                vasp_files.append(os.path.join(root, file))
            elif file_type == "OUTCAR" and file.endswith("OUTCAR"):
                vasp_files.append(os.path.join(root, file))
            elif file_type == "h5" and file.endswith("out.h5"):
                vasp_files.append(os.path.join(root, file))
    return vasp_files


def extract_config_origin(vasp_file):
    """
    Extract the configuration origin from the VASP file path.
    Returns the execution directory name joined with the relative path to the file.
    """
    exec_dir = os.path.basename(os.getcwd())
    rel_path = os.path.relpath(vasp_file)
    return os.path.join(exec_dir, rel_path)


def add_origin_to_atoms(atoms_list, origin):
    """Add the origin as a property to all atoms in the list."""
    for atoms in atoms_list:
        atoms.info['origin'] = origin


def renumber_ionic_steps(atoms_list):
    """Renumber ionic_step sequentially (1-based) for the given list of atoms."""
    for i, atoms in enumerate(atoms_list):
        atoms.info['ionic_step'] = i + 1


def relabel_atoms_file(input_file):
    """Change labels to REF_* format in the output file."""
    sed_cmd = [
        'sed', '-i',
        '-e', 's/forces/REF_forces/g',
        '-e', 's/stress/REF_stress/g',
        '-e', 's/free_energy/REF_energy/g',
        input_file
    ]
    subprocess.run(sed_cmd, check=True)


def save_configurations(configurations, output_file):
    """Save configurations to XYZ file."""
    with open(output_file, "w") as fout:
        for atoms in tqdm(configurations, desc=f"Saving {os.path.basename(output_file)}", disable=not HAS_TQDM):
            write(fout, atoms, format="extxyz")


def split_train_test(configurations, test_ratio=0.0, seed=123, shuffle=False):
    """Split configurations into train and test sets."""
    random.seed(seed)
    num_test = int(len(configurations) * test_ratio)
    configurations_copy = configurations.copy()
    if shuffle:
        random.shuffle(configurations_copy)
    test_configs = configurations_copy[:num_test]
    train_configs = configurations_copy[num_test:]
    return train_configs, test_configs


def save_split_and_report(all_configurations, train_file, test_file, args):
    """Split, save train/test files and print a summary."""
    train_configs, test_configs = split_train_test(
        all_configurations, test_ratio=args.test_ratio, seed=args.seed
    )
    print(f"  Configurations in the training set: {len(train_configs)}")
    print(f"  Configurations in the test set:     {len(test_configs)}")

    save_configurations(train_configs, train_file)
    save_configurations(test_configs, test_file)

    print(f"  Training file saved as '{train_file}'.")
    print(f"  Test file saved as '{test_file}'.")

    if args.relabel and not (args.file_type == "xml" and not args.no_enhanced):
        relabel_atoms_file(train_file)
        relabel_atoms_file(test_file)
        print("  Labels changed to REF_* in both output files.")


def process_first_last_mode(vasp_files, args, use_enhanced):
    """Process files to extract first and last N configurations."""
    all_configurations = []
    for vasp_file in tqdm(vasp_files, desc="Processing files", disable=not HAS_TQDM):
        try:
            atoms_list = load_atoms_from_file(vasp_file, use_enhanced=use_enhanced)
            origin = extract_config_origin(vasp_file)

            if len(atoms_list) <= 2 * args.num_configs:
                selected = atoms_list
            else:
                selected = atoms_list[:args.num_configs] + atoms_list[-args.num_configs:]

            add_origin_to_atoms(selected, origin)
            renumber_ionic_steps(selected)
            print(f"  {len(selected)} configurations selected (origin: {origin})")
            all_configurations.extend(selected)
        except Exception as e:
            print(f"Error processing file {vasp_file}: {e}")
    return all_configurations


def process_interval_mode(vasp_files, args, use_enhanced):
    """Process files to extract configurations at regular intervals."""
    all_configurations = []
    for vasp_file in tqdm(vasp_files, desc="Processing files", disable=not HAS_TQDM):
        try:
            atoms_list = load_atoms_from_file(vasp_file, use_enhanced=use_enhanced)
            origin = extract_config_origin(vasp_file)

            selected = atoms_list[::args.frame_interval]
            add_origin_to_atoms(selected, origin)
            renumber_ionic_steps(selected)
            print(f"  {len(selected)} configurations at interval {args.frame_interval} (origin: {origin})")
            all_configurations.extend(selected)
        except Exception as e:
            print(f"Error processing file {vasp_file}: {e}")
    return all_configurations


def process_all_mode(vasp_files, args, use_enhanced):
    """Process all configurations from all files."""
    all_configurations = []
    for vasp_file in tqdm(vasp_files, desc="Processing files", disable=not HAS_TQDM):
        try:
            atoms_list = load_atoms_from_file(vasp_file, use_enhanced=use_enhanced)
            origin = extract_config_origin(vasp_file)

            add_origin_to_atoms(atoms_list, origin)
            renumber_ionic_steps(atoms_list)
            print(f"  {len(atoms_list)} configurations (origin: {origin})")
            all_configurations.extend(atoms_list)
        except Exception as e:
            print(f"Error processing file {vasp_file}: {e}")
    return all_configurations


def process_first_frame_mode(vasp_files, args, use_enhanced):
    """Process files to extract only the first frame from each file."""
    all_configurations = []
    for vasp_file in tqdm(vasp_files, desc="Processing files", disable=not HAS_TQDM):
        try:
            atoms_list = load_atoms_from_file(vasp_file, use_enhanced=use_enhanced)
            origin = extract_config_origin(vasp_file)

            if atoms_list:
                frame = atoms_list[0]
                frame.info['origin'] = origin
                frame.info['ionic_step'] = 1
                all_configurations.append(frame)
                print(f"  First of {len(atoms_list)} frames selected (origin: {origin})")
            else:
                print(f"  Warning: No frames found in {vasp_file}")
        except Exception as e:
            print(f"Error processing file {vasp_file}: {e}")
    return all_configurations


def process_last_frame_mode(vasp_files, args, use_enhanced):
    """Process files to extract only the last frame from each file."""
    all_configurations = []
    for vasp_file in tqdm(vasp_files, desc="Processing files", disable=not HAS_TQDM):
        try:
            atoms_list = load_atoms_from_file(vasp_file, use_enhanced=use_enhanced)
            origin = extract_config_origin(vasp_file)

            if atoms_list:
                frame = atoms_list[-1]
                frame.info['origin'] = origin
                frame.info['ionic_step'] = 1
                all_configurations.append(frame)
                print(f"  Last of {len(atoms_list)} frames selected (origin: {origin})")
            else:
                print(f"  Warning: No frames found in {vasp_file}")
        except Exception as e:
            print(f"Error processing file {vasp_file}: {e}")
    return all_configurations


def main(args):
    vasp_files = find_vasp_files(args.data_path, args.file_type)

    if not vasp_files:
        print(f"No {args.file_type} files found in the path '{args.data_path}'.")
        return
    else:
        print(f"Detected files: {vasp_files}")

    use_enhanced = args.file_type == "xml" and not args.no_enhanced

    if use_enhanced:
        print("\n[INFO] Enhanced processing enabled (default for XML files):")
        print("  - Energy extraction with pymatgen (e_0_energy or e_fr_energy)")
        print("  - Forces and stress from ASE")
        print("  - Frame synchronization verification")
        print("  - Virial calculation")
        print("  - PSTRESS correction (when ISIF=3 and IBRION=0)")
        print("  - Output format: REF_energy, REF_forces, REF_stress, REF_virial")
        if args.relabel:
            print("  - Note: --relabel flag ignored (output already in REF_* format)")
        print()
    elif args.file_type == "xml":
        print("\n[INFO] Using legacy ASE-only processing (--no-enhanced flag detected)\n")

    os.makedirs(args.save_path, exist_ok=True)
    save_dir = args.save_path

    mode_funcs = {
        "first_last":   process_first_last_mode,
        "interval":     process_interval_mode,
        "all":          process_all_mode,
        "first_frame":  process_first_frame_mode,
        "last_frame":   process_last_frame_mode,
    }

    all_configurations = mode_funcs[args.mode](vasp_files, args, use_enhanced)

    if not all_configurations:
        print("No valid configurations found.")
        return

    print(f"\nTotal configurations collected: {len(all_configurations)}")

    if args.mode in ("first_frame", "last_frame"):
        filename = "first_frames.xyz" if args.mode == "first_frame" else "last_frames.xyz"
        output_file = os.path.join(save_dir, filename)
        save_configurations(all_configurations, output_file)
        print(f"  File saved as '{output_file}'.")
        if args.relabel and not use_enhanced:
            relabel_atoms_file(output_file)
            print("  Labels changed to REF_* in the output file.")
    elif args.mode == "interval":
        save_split_and_report(
            all_configurations,
            os.path.join(save_dir, "dataset-interval.xyz"),
            os.path.join(save_dir, "test-interval.xyz"),
            args,
        )
    else:  # first_last, all
        save_split_and_report(
            all_configurations,
            os.path.join(save_dir, "dataset.xyz"),
            os.path.join(save_dir, "test.xyz"),
            args,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified script for processing VASP files with different sampling strategies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract first and last 1000 configurations from each file
  %(prog)s --mode first_last --num_configs 1000 --data_path /path/to/vasp/files

  # Extract configurations at intervals of 500 with train/test split and relabeling
  %(prog)s --mode interval --frame_interval 500 --data_path /path/to/vasp/files --relabel

  # Process all configurations with 10%% test split
  %(prog)s --mode all --data_path /path/to/vasp/files --test_ratio 0.1

  # Extract only the first frame from each file
  %(prog)s --mode first_frame --data_path /path/to/vasp/files --save_path /path/to/output

  # Extract only the last frame from each file
  %(prog)s --mode last_frame --data_path /path/to/vasp/files --save_path /path/to/output

  # Use legacy ASE-only mode (disable enhanced processing)
  %(prog)s --mode all --data_path /path/to/vasp/files --no-enhanced

  # Enhanced processing is default for XML files
  %(prog)s --mode interval --frame_interval 500 --data_path /path/to/vasp/files
        """
    )

    parser.add_argument("--mode", "-m", type=str,
        choices=["first_last", "interval", "all", "first_frame", "last_frame"],
        required=True,
        help="Processing mode.")
    parser.add_argument("--data_path", "-dp", type=str, default='.',
        help="Path to the directory containing VASP files.")
    parser.add_argument("--save_path", "-sp", type=str, default='.',
        help="Path to save the processed data.")
    parser.add_argument("--file_type", "-ft", type=str,
        choices=["xml", "OUTCAR", "h5"], default="xml",
        help="Type of VASP files to process.")
    parser.add_argument("--seed", type=int, default=123,
        help="Seed for randomness in train/test split.")
    parser.add_argument("--test_ratio", "-tr", type=float, default=0.0,
        help="Fraction of configurations to use as test set (default: 0.0).")
    parser.add_argument("--num_configs", "-nc", type=int, default=1000,
        help="Configurations to take from start and end of each file (first_last mode).")
    parser.add_argument("--frame_interval", "-fi", type=int, default=500,
        help="Interval between frames to extract (interval mode).")
    parser.add_argument("--relabel", "-r", action="store_true",
        help="Apply relabeling to REF_* format (forces, stress, free_energy).")
    parser.add_argument("--no-enhanced", "-ne", action="store_true",
        help="Disable enhanced XML processing; use legacy ASE-only mode.")

    args = parser.parse_args()
    main(args)
