#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze SiO2 geometrical properties: Si-O distances and O-Si-O / Si-O-Si angles
for each configuration in an XYZ file.

Usage:
    python analyze_sio2_geometry.py <xyz_file> --rmin <value> --rmax <value>

Output:
    Creates <config_number>conf.txt files with geometry statistics for each configuration
"""

import argparse
import numpy as np
import os
from ase.io import read
from ase import Atoms


def calculate_angle(v1, v2):
    """
    Calculate angle between two vectors in degrees.
    
    Args:
        v1, v2: numpy arrays representing vectors
        
    Returns:
        Angle in degrees
    """
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # Clip to avoid numerical errors with arccos
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def check_complete_tetrahedra(atoms, rmin, rmax):
    """
    Check if there are incomplete or over-coordinated SiO4 tetrahedra within the radius range.
    A proper tetrahedron has exactly 4 O neighbors within rmin-rmax.
    Dangerous = Si atom with != 4 O neighbors.
    Uses periodic boundary conditions (PBC) via minimum image convention.
    
    Args:
        atoms: ASE Atoms object
        rmin: Minimum radius (Angstroms)
        rmax: Maximum radius (Angstroms)
        
    Returns:
        Boolean indicating if dangerous configurations found, and list of Si indices with incomplete tetrahedra
    """
    symbols = np.array(atoms.get_chemical_symbols())
    
    si_indices = np.where(symbols == 'Si')[0]
    o_indices = np.where(symbols == 'O')[0]
    
    if len(si_indices) == 0 or len(o_indices) == 0:
        return False, []
    
    dangerous_si = []
    
    for si_idx in si_indices:
        # Calculate distances using minimum image convention (PBC aware)
        distances = atoms.get_distances(si_idx, o_indices, mic=True, vector=False)
        
        # Count valid O neighbors within rmin-rmax
        n_neighbors = np.sum((distances >= rmin) & (distances <= rmax))
        
        # If NOT exactly 4 O neighbors, it's dangerous (incomplete or over-coordinated)
        if n_neighbors != 4:
            dangerous_si.append(si_idx)
    
    return len(dangerous_si) > 0, dangerous_si


def check_small_cell(cell_lengths, threshold=10.0):
    """
    Check if any cell parameter is smaller than threshold.
    
    Args:
        cell_lengths: Array with cell lengths [a, b, c]
        threshold: Minimum acceptable cell length (Angstroms, default: 10.0)
        
    Returns:
        Boolean indicating if cell is too small, list of small parameters
    """
    small_params = []
    param_names = ['a', 'b', 'c']
    
    for i, (length, name) in enumerate(zip(cell_lengths, param_names)):
        if length < threshold:
            small_params.append((name, length))
    
    return len(small_params) > 0, small_params


def analyze_sio2_structure(atoms, rmin, rmax):
    """
    Analyze SiO2 structure: Si-O distances and angles.
    Uses periodic boundary conditions (PBC) via minimum image convention.
    
    Args:
        atoms: ASE Atoms object
        rmin: Minimum radius for neighbor search (Angstroms)
        rmax: Maximum radius for neighbor search (Angstroms)
        
    Returns:
        Dictionary with statistics: distances, OSiO angles, SiOSi angles, SiSiSi angles, Si count, volume, density
    """
    symbols = np.array(atoms.get_chemical_symbols())
    
    # Get indices of Si and O atoms
    si_indices = np.where(symbols == 'Si')[0]
    o_indices = np.where(symbols == 'O')[0]
    
    if len(si_indices) == 0:
        return None
    
    # Calculate cell volume and Si density
    cell = atoms.get_cell()
    cell_lengths = cell.lengths()  # a, b, c
    cell_angles = cell.angles()    # alpha, beta, gamma
    volume = atoms.get_volume()
    n_si = len(si_indices)
    si_density = n_si / volume if volume > 0 else 0.0
    
    # Check for complete tetrahedra (dangerous configurations)
    has_tetra_danger, dangerous_si_indices = check_complete_tetrahedra(atoms, rmin, rmax)
    
    # Check for small cell parameters (dangerous configurations)
    has_cell_danger, small_cell_params = check_small_cell(cell_lengths)
    
    # Overall danger status
    has_danger = has_tetra_danger or has_cell_danger
    
    # Storage for measurements
    sio_distances = []
    osio_angles = []
    siosi_angles = []
    sisisi_angles = []
    
    if len(o_indices) > 0:
        # For each Si atom, find O neighbors and calculate O-Si-O angles
        for si_idx in si_indices:
            # Calculate distances using PBC
            distances = atoms.get_distances(si_idx, o_indices, mic=True, vector=False)
            vectors = atoms.get_distances(si_idx, o_indices, mic=True, vector=True)
            
            # Filter by rmin-rmax
            valid_mask = (distances >= rmin) & (distances <= rmax)
            valid_distances = distances[valid_mask]
            valid_vectors = vectors[valid_mask]
            valid_o_indices = o_indices[valid_mask]
            
            # Store Si-O distances
            sio_distances.extend(valid_distances)
            
            # Calculate O-Si-O angles for this Si
            if len(valid_o_indices) >= 2:
                for i in range(len(valid_o_indices)):
                    for j in range(i + 1, len(valid_o_indices)):
                        v1 = valid_vectors[i]
                        v2 = valid_vectors[j]
                        angle = calculate_angle(v1, v2)
                        osio_angles.append(angle)
        
        # For each O atom, find Si neighbors and calculate Si-O-Si angles
        for o_idx in o_indices:
            # Calculate distances using PBC
            distances = atoms.get_distances(o_idx, si_indices, mic=True, vector=False)
            vectors = atoms.get_distances(o_idx, si_indices, mic=True, vector=True)
            
            # Filter by rmin-rmax
            valid_mask = (distances >= rmin) & (distances <= rmax)
            valid_si_indices = si_indices[valid_mask]
            valid_vectors = vectors[valid_mask]
            
            # Calculate Si-O-Si angles for this O
            if len(valid_si_indices) >= 2:
                for i in range(len(valid_si_indices)):
                    for j in range(i + 1, len(valid_si_indices)):
                        v1 = valid_vectors[i]
                        v2 = valid_vectors[j]
                        angle = calculate_angle(v1, v2)
                        siosi_angles.append(angle)
    
    # For each Si atom, find Si neighbors and calculate Si-Si-Si angles
    # Use larger radius for Si-Si interactions
    rmax_sisi = rmax * 3.5
    for si_idx in si_indices:
        # Calculate distances using PBC
        distances = atoms.get_distances(si_idx, si_indices, mic=True, vector=False)
        vectors = atoms.get_distances(si_idx, si_indices, mic=True, vector=True)
        
        # Filter by distance and exclude self
        valid_mask = (distances > 0) & (distances <= rmax_sisi)
        valid_si_indices = si_indices[valid_mask]
        valid_vectors = vectors[valid_mask]
        
        # Calculate Si-Si-Si angles for this Si (central Si)
        if len(valid_si_indices) >= 2:
            for i in range(len(valid_si_indices)):
                for j in range(i + 1, len(valid_si_indices)):
                    v1 = valid_vectors[i]
                    v2 = valid_vectors[j]
                    angle = calculate_angle(v1, v2)
                    sisisi_angles.append(angle)
    
    # Calculate statistics
    result = {
        'n_si': n_si,
        'volume': volume,
        'si_density': si_density,
        'cell_lengths': cell_lengths,
        'cell_angles': cell_angles,
        'has_danger': has_danger,
        'has_tetra_danger': has_tetra_danger,
        'has_cell_danger': has_cell_danger,
        'n_dangerous_tetrahedra': len(dangerous_si_indices),
        'small_cell_params': small_cell_params
    }
    
    if len(sio_distances) > 0:
        result['sio_distance'] = {
            'min': np.min(sio_distances),
            'mean': np.mean(sio_distances),
            'max': np.max(sio_distances)
        }
    else:
        result['sio_distance'] = None
    
    if len(siosi_angles) > 0:
        result['siosi_angle'] = {
            'min': np.min(siosi_angles),
            'mean': np.mean(siosi_angles),
            'max': np.max(siosi_angles)
        }
    else:
        result['siosi_angle'] = None
    
    if len(osio_angles) > 0:
        result['osio_angle'] = {
            'min': np.min(osio_angles),
            'mean': np.mean(osio_angles),
            'max': np.max(osio_angles)
        }
    else:
        result['osio_angle'] = None
    
    if len(sisisi_angles) > 0:
        result['sisisi_angle'] = {
            'min': np.min(sisisi_angles),
            'mean': np.mean(sisisi_angles),
            'max': np.max(sisisi_angles)
        }
    else:
        result['sisisi_angle'] = None
    
    return result


def write_output_file(filename, config_num, stats, xyz_filename, origin=None):
    """
    Write statistics to output file.
    
    Args:
        filename: Output filename
        config_num: Configuration number
        stats: Dictionary with statistics
        xyz_filename: Name of the analyzed XYZ file
        origin: Origin from atoms.info if available
    """
    with open(filename, 'w') as f:
        f.write(f"File:           {origin if origin else xyz_filename}\n")
        f.write(f"Configuration:  {config_num}\n")
        f.write("=" * 50 + "\n\n")
        
        # Warning if dangerous configuration
        if stats['has_danger']:
            f.write("⚠️  OJO: Esta configuración es PELIGROSA\n")
            
            if stats['has_tetra_danger']:
                f.write(f"    • TETRAEDROS INCORRECTOS: Se encontraron {stats['n_dangerous_tetrahedra']} átomos de Si\n")
                f.write("      con coordinación diferente a 4 (incompletos o sobre-coordinados).\n")
            
            if stats['has_cell_danger']:
                f.write("    • PARÁMETROS DE CELDA PEQUEÑOS (< 10 Å):\n")
                for param_name, param_value in stats['small_cell_params']:
                    f.write(f"      {param_name} = {param_value:.6f} Å\n")
            
            f.write("\n")
        
        # Si atoms, volume and density
        f.write(f"Si_atoms:       {stats['n_si']}\n")
        f.write(f"Volume_cell:    {stats['volume']:.6f} Å³\n")
        f.write(f"Si_density:     {stats['si_density']:.6f} atoms/Å³\n\n")
        
        # Cell parameters
        f.write("Cell parameters:\n")
        f.write(f"  a = {stats['cell_lengths'][0]:.6f} Å\n")
        f.write(f"  b = {stats['cell_lengths'][1]:.6f} Å\n")
        f.write(f"  c = {stats['cell_lengths'][2]:.6f} Å\n")
        f.write(f"  α = {stats['cell_angles'][0]:.6f} °\n")
        f.write(f"  β = {stats['cell_angles'][1]:.6f} °\n")
        f.write(f"  γ = {stats['cell_angles'][2]:.6f} °\n\n")
        
        f.write("                Min        Mean       Max\n")
        f.write("-" * 50 + "\n")
        
        # Si-O distances
        if stats['sio_distance'] is not None:
            d = stats['sio_distance']
            f.write(f"SiO_distance    {d['min']:<10.6f} {d['mean']:<10.6f} {d['max']:<10.6f}\n")
        else:
            f.write("SiO_distance    No data\n")
        
        # Si-O-Si angles
        if stats['siosi_angle'] is not None:
            a = stats['siosi_angle']
            f.write(f"SiOSi_angle     {a['min']:<10.6f} {a['mean']:<10.6f} {a['max']:<10.6f}\n")
        else:
            f.write("SiOSi_angle     No data\n")
        
        # O-Si-O angles
        if stats['osio_angle'] is not None:
            a = stats['osio_angle']
            f.write(f"OSiO_angle      {a['min']:<10.6f} {a['mean']:<10.6f} {a['max']:<10.6f}\n")
        else:
            f.write("OSiO_angle      No data\n")
        
        # Si-Si-Si angles
        if stats['sisisi_angle'] is not None:
            a = stats['sisisi_angle']
            f.write(f"SiSiSi_angle    {a['min']:<10.6f} {a['mean']:<10.6f} {a['max']:<10.6f}\n")
        else:
            f.write("SiSiSi_angle    No data\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze SiO2 geometrical properties from XYZ file',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('xyz_file', type=str, help='Path to XYZ file')
    parser.add_argument('--rmin', type=float, default=1.4, 
                        help='Minimum radius for neighbor search (Angstroms, default: 1.4)')
    parser.add_argument('--rmax', type=float, default=2.0,
                        help='Maximum radius for neighbor search (Angstroms, default: 2.0)')
    
    args = parser.parse_args()
    
    # Read all configurations
    print(f"Reading configurations from {args.xyz_file}...")
    configurations = read(args.xyz_file, index=':')
    
    # Handle single configuration
    if not isinstance(configurations, list):
        configurations = [configurations]
    
    print(f"Found {len(configurations)} configurations")
    print(f"Using rmin={args.rmin} Å, rmax={args.rmax} Å\n")
    
    # Create output directory
    output_dir = "confs"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}/\n")
    
    # Calculate progress milestones (every 10%)
    total_configs = len(configurations)
    progress_step = max(1, total_configs // 10)
    
    # Process each configuration
    for i, atoms in enumerate(configurations):
        config_num = i + 1
        
        # Print progress every 10%
        if config_num % progress_step == 0 or config_num == 1 or config_num == total_configs:
            percentage = (config_num / total_configs) * 100
            print(f"Processing: {config_num}/{total_configs} ({percentage:.0f}%)")
        
        # Analyze structure
        stats = analyze_sio2_structure(atoms, args.rmin, args.rmax)
        
        if stats is None:
            print(f"  ⚠️  Configuration {config_num}: No Si atoms found!")
            continue
        
        # Get origin from atoms.info if available
        origin = atoms.info.get('origin', None)
        
        # Write output file
        output_filename = os.path.join(output_dir, f"{config_num}conf.txt")
        write_output_file(output_filename, config_num, stats, args.xyz_file, origin)
    
    print(f"\n✓ Analysis complete! Processed {len(configurations)} configurations.")


if __name__ == "__main__":
    main()
