#!/usr/bin/env python3
"""
Convert XYZ files to LAMMPS data files compatible with SLC (Sanders-Leslie-Catlow) force field.
Richard Catlow et al. force field for zeolites.

Usage:
    python xyz_to_lammps_slc.py input.xyz output.data
"""

import sys
import os
import numpy as np
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs


def assign_atom_types_and_charges(atoms):
    """
    Assign LAMMPS atom types and charges for core-shell model.
    
    Core-shell model for zeolites (SLC-compatible):
    - Type 1: Si core (+4.0 |e|)
    - Type 2: O core (+0.8690 |e|)
    - Type 3: O shell (-2.8690 |e|, mass 0.1)
    
    For each O atom, creates a core and a shell at the same position.
    
    Returns:
        atom_types: list of LAMMPS atom type integers
        charges: list of charges
        type_names: dict mapping type int to type name
        n_cores: number of core atoms (Si + O_core)
    """
    symbols = atoms.get_chemical_symbols()
    n_atoms = len(atoms)
    
    atom_types = []
    charges = []
    positions_list = []
    
    # Process each atom
    for i, sym in enumerate(symbols):
        pos = atoms.positions[i]
        
        if sym == 'Si':
            # Si core only
            atom_types.append(1)
            charges.append(4.0)
            positions_list.append(pos)
            
        elif sym == 'O':
            # O core
            atom_types.append(2)
            charges.append(0.8690)
            positions_list.append(pos)
            
            # O shell (same position)
            atom_types.append(3)
            charges.append(-2.8690)
            positions_list.append(pos)
    
    type_names = {
        1: 'Si',
        2: 'O_core',
        3: 'O_shell'
    }
    
    # Count cores (Si + O_core)
    n_si = symbols.count('Si')
    n_o = symbols.count('O')
    n_cores = n_si + n_o
    
    return atom_types, charges, np.array(positions_list), type_names, n_cores


def find_bonds(atoms, n_cores):
    """
    Generate core-shell bonds for oxygen atoms only.
    Each O atom has a bond between its core (type 2) and shell (type 3).
    
    Args:
        atoms: ASE atoms object (original, before shell duplication)
        n_cores: number of core atoms
        
    Returns:
        bonds: list of (atom_i, atom_j, bond_type=1) tuples
    """
    symbols = atoms.get_chemical_symbols()
    bonds = []
    
    # Track atom indices in the expanded system
    atom_idx = 1  # LAMMPS 1-indexed
    
    for sym in symbols:
        if sym == 'Si':
            atom_idx += 1  # Si core only
        elif sym == 'O':
            o_core_idx = atom_idx
            o_shell_idx = atom_idx + 1
            bonds.append((o_core_idx, o_shell_idx, 1))  # Bond type 1: O_core-O_shell
            atom_idx += 2  # O core + O shell
    
    bond_types = {('O_core', 'O_shell'): 1}
    
    return bonds, bond_types


def read_forcefield_lib(filename='forcefield.lib'):
    """
    Read force field parameters from forcefield.lib file.
    Supports RASPA format with atom type names.
    
    Expected format:
    Bond Coeffs   RASPA
    atom1 atom2 style param1 param2 ...
    
    Angle Coeffs  RASPA  
    atom1 atom2 atom3 style param1 param2 ...
    
    Pair Coeffs   RASPA
    atom1 atom2 style param1 param2 ...
    
    Masses (optional)
    atom mass
    
    Returns:
        dict with keys: 'masses', 'bond_coeffs', 'angle_coeffs', 'dihedral_coeffs',
                       'improper_coeffs', 'pair_coeffs', 'atom_type_map'
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Force field file '{filename}' not found in current directory")
    
    ff_params = {
        'masses': {},  # atom_name -> mass
        'bond_coeffs': [],  # (atom1, atom2, style, [params])
        'angle_coeffs': [],  # (atom1, atom2, atom3, style, [params])
        'dihedral_coeffs': [],  # (a1, a2, a3, a4, style, [params])
        'improper_coeffs': [],  # (a1, a2, a3, a4, style, [params])
        'pair_coeffs': [],  # (atom1, atom2, style, [params])
        'atom_type_map': {}  # atom_name -> type_id (will be built)
    }
    
    current_section = None
    
    with open(filename, 'r') as f:
        for line in f:
            # Remove comments and strip whitespace
            line = line.split('#')[0].strip()
            
            if not line:
                continue
            
            # Check for end marker
            if line.lower() == 'end':
                break
            
            # Check for section headers (case insensitive, allow RASPA suffix)
            line_upper = line.upper()
            if 'MASSES' in line_upper:
                current_section = 'masses'
                continue
            elif 'BOND COEFFS' in line_upper or 'BOND' in line_upper and 'COEFFS' in line_upper:
                current_section = 'bond_coeffs'
                continue
            elif 'ANGLE COEFFS' in line_upper or 'ANGLE' in line_upper and 'COEFFS' in line_upper:
                current_section = 'angle_coeffs'
                continue
            elif 'DIHEDRAL COEFFS' in line_upper or 'DIHEDRAL' in line_upper and 'COEFFS' in line_upper:
                current_section = 'dihedral_coeffs'
                continue
            elif 'IMPROPER COEFFS' in line_upper or 'IMPROPER' in line_upper and 'COEFFS' in line_upper:
                current_section = 'improper_coeffs'
                continue
            elif 'PAIR COEFFS' in line_upper or ('PAIR' in line_upper and 'COEFFS' in line_upper):
                current_section = 'pair_coeffs'
                continue
            
            # Parse data lines
            if current_section:
                parts = line.split()
                if not parts:
                    continue
                
                try:
                    if current_section == 'masses':
                        # Format: atom_name mass
                        atom_name = parts[0]
                        mass = float(parts[1])
                        ff_params['masses'][atom_name] = mass
                        
                    elif current_section == 'bond_coeffs':
                        # Format: atom1 atom2 style param1 param2 ...
                        atom1 = parts[0]
                        atom2 = parts[1]
                        style = parts[2]
                        params = [float(p) for p in parts[3:]]
                        ff_params['bond_coeffs'].append((atom1, atom2, style, params))
                        
                    elif current_section == 'angle_coeffs':
                        # Format: atom1 atom2 atom3 style param1 param2 ...
                        atom1 = parts[0]
                        atom2 = parts[1]
                        atom3 = parts[2]
                        style = parts[3]
                        params = [float(p) for p in parts[4:]]
                        ff_params['angle_coeffs'].append((atom1, atom2, atom3, style, params))
                        
                    elif current_section == 'dihedral_coeffs':
                        # Format: atom1 atom2 atom3 atom4 style param1 param2 ...
                        atom1 = parts[0]
                        atom2 = parts[1]
                        atom3 = parts[2]
                        atom4 = parts[3]
                        style = parts[4]
                        params = [float(p) for p in parts[5:]]
                        ff_params['dihedral_coeffs'].append((atom1, atom2, atom3, atom4, style, params))
                        
                    elif current_section == 'improper_coeffs':
                        # Format: atom1 atom2 atom3 atom4 style param1 param2 ...
                        atom1 = parts[0]
                        atom2 = parts[1]
                        atom3 = parts[2]
                        atom4 = parts[3]
                        style = parts[4]
                        params = [float(p) for p in parts[5:]]
                        ff_params['improper_coeffs'].append((atom1, atom2, atom3, atom4, style, params))
                        
                    elif current_section == 'pair_coeffs':
                        # Format: atom1 atom2 style param1 param2 ...
                        atom1 = parts[0]
                        atom2 = parts[1]
                        style = parts[2]
                        params = [float(p) for p in parts[3:]]
                        ff_params['pair_coeffs'].append((atom1, atom2, style, params))
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse line in {current_section}: {line}")
                    continue
    
    return ff_params


def find_angles(atoms, n_cores):
    """
    Generate angles for the core-shell model.
    
    For each Si-O pair, creates angles:
    - Type 1: O_shell - Si - O_shell
    - Type 2: Si - O_shell - Si (dummy, K=0)
    - Type 3: Si - O_shell - O_core (dummy, K=0)
    
    Args:
        atoms: ASE atoms object (original)
        n_cores: number of core atoms
        
    Returns:
        angles: list of (atom_i, atom_j, atom_k, angle_type) tuples
        angle_types: dict of angle type assignments
    """
    # Use neighbor list to find Si-O connectivity
    cutoffs = natural_cutoffs(atoms, mult=1.3)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)
    
    symbols = atoms.get_chemical_symbols()
    angles = []
    
    # Map original atom index to expanded system indices
    original_to_expanded = {}
    atom_idx = 1
    for i, sym in enumerate(symbols):
        if sym == 'Si':
            original_to_expanded[i] = {'core': atom_idx}
            atom_idx += 1
        elif sym == 'O':
            original_to_expanded[i] = {'core': atom_idx, 'shell': atom_idx + 1}
            atom_idx += 2
    
    # Find angles: O_shell - Si - O_shell
    for i, sym in enumerate(symbols):
        if sym == 'Si':
            si_idx = original_to_expanded[i]['core']
            indices, _ = nl.get_neighbors(i)
            
            o_neighbors = [j for j in indices if symbols[j] == 'O']
            
            # Create angles between pairs of O shells bonded to the same Si
            for idx1 in range(len(o_neighbors)):
                for idx2 in range(idx1 + 1, len(o_neighbors)):
                    o1 = o_neighbors[idx1]
                    o2 = o_neighbors[idx2]
                    
                    o1_shell = original_to_expanded[o1]['shell']
                    o2_shell = original_to_expanded[o2]['shell']
                    
                    # Angle type 1: O_shell - Si - O_shell
                    angles.append((o1_shell, si_idx, o2_shell, 1))
    
    angle_types = {
        ('O_shell', 'Si', 'O_shell'): 1,
        ('Si', 'O_shell', 'Si'): 2,  # Dummy
        ('Si', 'O_shell', 'O_core'): 3  # Dummy
    }
    
    return angles, angle_types


def write_lammps_data(filename, atoms, atom_types, charges, positions, bonds, angles, ff_params):
    """
    Write LAMMPS data file in 'full' atom style for core-shell model.
    
    Args:
        filename: output file name
        atoms: ASE atoms object (original)
        atom_types: list of atom type integers
        charges: list of charges
        positions: numpy array of positions (includes shells)
        bonds: list of (i, j, type) tuples
        angles: list of (i, j, k, type) tuples
        ff_params: dict with force field parameters from forcefield.lib
    """
    n_atoms = len(atom_types)
    n_bonds = len(bonds)
    n_angles = len(angles)
    
    n_atom_types = 3  # Si, O_core, O_shell
    n_bond_types = 1  # O_core - O_shell
    n_angle_types = 3  # shell-Si-shell, Si-shell-Si, Si-shell-core
    
    cell = atoms.get_cell()
    
    # Get cell parameters
    xlo, ylo, zlo = 0.0, 0.0, 0.0
    xhi = cell[0, 0]
    yhi = cell[1, 1]
    zhi = cell[2, 2]
    xy = cell[1, 0]
    xz = cell[2, 0]
    yz = cell[2, 1]
    
    with open(filename, 'w') as f:
        f.write(f"LAMMPS data file with core-shell model (SLC force field)\n\n")
        
        # Header
        f.write(f"{n_atoms} atoms\n")
        f.write(f"{n_bonds} bonds\n")
        f.write(f"{n_angles} angles\n")
        f.write("0 dihedrals\n")
        f.write("0 impropers\n\n")
        
        f.write(f"{n_atom_types} atom types\n")
        f.write(f"{n_bond_types} bond types\n")
        f.write(f"{n_angle_types} angle types\n")
        f.write("\n")
        
        # Box
        f.write(f"{xlo:.10f} {xhi:.10f} xlo xhi\n")
        f.write(f"{ylo:.10f} {yhi:.10f} ylo yhi\n")
        f.write(f"{zlo:.10f} {zhi:.10f} zlo zhi\n")
        
        if abs(xy) > 1e-6 or abs(xz) > 1e-6 or abs(yz) > 1e-6:
            f.write(f"{xy:.10f} {xz:.10f} {yz:.10f} xy xz yz\n")
        f.write("\n")
        
        # Masses - use default values for Si, O_core, O_shell
        f.write("Masses\n\n")
        default_masses = {
            1: ('Si', 28.0855),
            2: ('O_core', 15.9994),
            3: ('O_shell', 0.1)
        }
        
        for type_id in sorted(default_masses.keys()):
            atom_name, mass = default_masses[type_id]
            # Override with value from forcefield.lib if available
            if atom_name in ff_params['masses']:
                mass = ff_params['masses'][atom_name]
            elif 'shO2' in ff_params['masses'] and atom_name == 'O_shell':
                mass = ff_params['masses'].get('shO2', mass)
            f.write(f"{type_id} {mass:.6f}  # {atom_name}\n")
        f.write("\n")
        
        # Bond Coeffs - filter for O_core-O_shell bonds
        f.write("Bond Coeffs\n\n")
        # Look for shO2-O2 or similar core-shell bonds
        bond_found = False
        for atom1, atom2, style, params in ff_params['bond_coeffs']:
            # Look for O core-shell bond (various naming conventions)
            if ('shO2' in atom1 and 'O2' in atom2) or ('O2' in atom1 and 'shO2' in atom2):
                params_str = ' '.join([f"{p:.6f}" for p in params])
                f.write(f"1 {params_str}  # {style} O_core - O_shell\n")
                bond_found = True
                break
        
        if not bond_found:
            # Use default values
            f.write("1 37.46 0.0  # harmonic O_core - O_shell (default)\n")
        f.write("\n")
        
        # Angle Coeffs
        f.write("Angle Coeffs\n\n")
        # Look for relevant angles involving Si and O shells
        angles_written = {1: False, 2: False, 3: False}
        
        for atom1, atom2, atom3, style, params in ff_params['angle_coeffs']:
            # Type 1: O_shell - Si - O_shell
            if not angles_written[1] and 'shO2' in atom1 and 'Si' in atom2 and 'shO2' in atom3:
                params_str = ' '.join([f"{p:.6f}" for p in params])
                f.write(f"1 {params_str}  # {style} O_shell - Si - O_shell\n")
                angles_written[1] = True
        
        # Write defaults for missing angles
        if not angles_written[1]:
            f.write("1 1.0486 109.47  # harmonic O_shell - Si - O_shell (default)\n")
        f.write("2 0.0 0.0        # Si - O_shell - Si (dummy)\n")
        f.write("3 0.0 0.0        # Si - O_shell - O_core (dummy)\n")
        f.write("\n")
        
        # PairIJ Coeffs - for buck/coul/long style
        # Format: type_i type_j A rho C (NO style name in data file)
        f.write("PairIJ Coeffs\n\n")
        # Map relevant pairs from forcefield.lib
        pair_map = {
            ('Si1', 'Si1'): (1, 1),
            ('Si2', 'Si2'): (1, 1),
            ('Si1', 'Si2'): (1, 1),
            ('Si1', 'shO2'): (1, 3),
            ('Si2', 'shO2'): (1, 3),
            ('shO2', 'shO2'): (3, 3),
        }
        
        pairs_written = set()
        
        for atom1, atom2, style, params in ff_params['pair_coeffs']:
            key = (atom1, atom2)
            if key in pair_map:
                type_i, type_j = pair_map[key]
                pair_key = (min(type_i, type_j), max(type_i, type_j))
                
                if pair_key not in pairs_written:
                    # Only write parameters, NOT the style
                    params_str = ' '.join([f"{p:.6f}" for p in params])
                    f.write(f"{type_i} {type_j} {params_str}\n")
                    pairs_written.add(pair_key)
        
        # Write defaults for missing pairs (Buckingham: A=0, rho=1, C=0 for dummy)
        default_pairs = [
            (1, 1, [0.0, 1.0, 0.0], 'Si - Si'),
            (1, 2, [0.0, 1.0, 0.0], 'Si - O_core'),
            (2, 2, [0.0, 1.0, 0.0], 'O_core - O_core'),
            (2, 3, [0.0, 1.0, 0.0], 'O_core - O_shell'),
        ]
        
        for type_i, type_j, params, comment in default_pairs:
            pair_key = (min(type_i, type_j), max(type_i, type_j))
            if pair_key not in pairs_written:
                params_str = ' '.join([f"{p:.6f}" for p in params])
                f.write(f"{type_i} {type_j} {params_str}  # {comment} (default)\n")
        
        f.write("\n")
        
        # Atoms
        f.write("Atoms # full\n\n")
        for i in range(n_atoms):
            mol_id = 1
            atom_type = atom_types[i]
            charge = charges[i]
            x, y, z = positions[i]
            
            # Add comment for clarity
            if atom_type == 1:
                comment = "# Si"
            elif atom_type == 2:
                comment = "# O_core"
            else:
                comment = "# O_shell"
            
            f.write(f"{i+1} {mol_id} {atom_type} {charge:.6f} {x:.10f} {y:.10f} {z:.10f} {comment}\n")
        f.write("\n")
        
        # Bonds
        if bonds:
            f.write("Bonds\n\n")
            for idx, (i, j, bond_type) in enumerate(bonds):
                f.write(f"{idx+1} {bond_type} {i} {j}\n")
            f.write("\n")
        
        # Angles
        if angles:
            f.write("Angles\n\n")
            for idx, (i, j, k, angle_type) in enumerate(angles):
                f.write(f"{idx+1} {angle_type} {i} {j} {k}\n")
            f.write("\n")
    
    print(f"LAMMPS data file written: {filename}")
    print(f"  Total atoms: {n_atoms}")
    print(f"  Bonds (core-shell): {n_bonds}")
    print(f"  Angles: {n_angles}")


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python xyz_to_lammps_slc.py input.xyz [output.data]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # If output file not provided, generate it from input file
    if len(sys.argv) == 3:
        output_file = sys.argv[2]
    else:
        # Replace .xyz extension with .data
        if input_file.lower().endswith('.xyz'):
            output_file = input_file[:-4] + '.data'
        else:
            output_file = input_file + '.data'
    
    # Read force field parameters
    print("Reading force field parameters from forcefield.lib...")
    try:
        ff_params = read_forcefield_lib('forcefield.lib')
        print(f"  Masses: {len(ff_params['masses'])} atom types")
        print(f"  Bond coeffs: {len(ff_params['bond_coeffs'])} entries")
        print(f"  Angle coeffs: {len(ff_params['angle_coeffs'])} entries")
        print(f"  Pair coeffs: {len(ff_params['pair_coeffs'])} pairs")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please create a forcefield.lib file in the current directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading forcefield.lib: {e}")
        sys.exit(1)
    
    print(f"\nReading structure from: {input_file}")
    atoms = read(input_file)
    
    print(f"Structure contains {len(atoms)} atoms")
    print(f"Composition: {atoms.get_chemical_formula()}")
    
    # Assign atom types and charges (creates core-shell pairs for O)
    atom_types, charges, positions, type_names, n_cores = assign_atom_types_and_charges(atoms)
    
    n_si = atoms.get_chemical_symbols().count('Si')
    n_o = atoms.get_chemical_symbols().count('O')
    
    print(f"\nCore-shell model created:")
    print(f"  Si atoms: {n_si}")
    print(f"  O atoms: {n_o} -> {n_o} cores + {n_o} shells")
    print(f"  Total atoms in LAMMPS: {len(atom_types)}")
    
    # Find bonds (core-shell bonds for O)
    print("\nGenerating core-shell bonds...")
    bonds, bond_types = find_bonds(atoms, n_cores)
    print(f"  Bonds: {len(bonds)} (O_core - O_shell)")
    
    # Find angles
    print("\nGenerating angles...")
    angles, angle_types = find_angles(atoms, n_cores)
    print(f"  Angles: {len(angles)}")
    
    # Write LAMMPS data file
    print(f"\nWriting LAMMPS data file: {output_file}")
    write_lammps_data(output_file, atoms, atom_types, charges, positions, bonds, angles, ff_params)
    
    print("\nDone!")
    print("\nNote: The SLC force field parameters are already included in the data file.")
    print("Use in your LAMMPS input:")
    print("  pair_style buck/coul/long 12.0")
    print("  bond_style harmonic")
    print("  angle_style harmonic")
    print("  kspace_style pppm 1e-5")
    print("\nNote: PairIJ Coeffs in data file uses Buckingham parameters (A, rho, C)")


if __name__ == "__main__":
    main()
