#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to filter atomic configurations based on different properties.
Can filter by energy per atom, total energy, forces, and stress.
"""
import sys
from ase.io import read, write
import numpy as np
import os
from tqdm import tqdm


def write_filtered_frames(output_file, valid_frames):
    """
    Helper function to write filtered frames to a file.
    
    Args:
        output_file (str): Path to output file
        valid_frames (list): List of valid frames
    """
    if not valid_frames:
        return False
        
    with open(output_file, 'w') as f_out:
        for i, atoms in enumerate(tqdm(valid_frames, desc="Writing frames", unit="frame")):
            if i == 0:
                write(f_out, atoms, format='extxyz')
            else:
                write(f_out, atoms, format='extxyz', append=True)
    return True


def print_statistics(property_name, input_file, output_file, threshold, 
                    total_frames, kept_frames, units, extra_info=None):
    """
    Helper function to print filtering statistics.
    
    Args:
        property_name (str): Name of the filtered property
        input_file (str): Input file path
        output_file (str): Output file path
        threshold (float): Threshold used
        total_frames (int): Total processed frames
        kept_frames (int): Frames kept after filtering
        units (str): Property units
        extra_info (dict): Additional property-specific information
    """
    print(f"\n{'='*50}")
    print(f"{property_name.upper()} FILTERING STATISTICS")
    print("="*50)
    
    if extra_info:
        print(f"Total configurations read: {total_frames}")
        # Only show if keys exist (energy case)
        if 'below' in extra_info:
            print(f"Configurations below threshold ({threshold} {units}): {extra_info['below']}")
        if 'above' in extra_info:
            print(f"Configurations above threshold ({threshold} {units}): {extra_info['above']}")
        if 'group_type' in extra_info:
            print(f"Largest group saved: {extra_info['group_type']} threshold")
        print(f"✓ Main file saved: {output_file}")
        print(f"Total configurations saved: {kept_frames}")
        # Show information about discarded file
        if extra_info.get('discarded_filename'):
            print(f"✓ Discarded file saved: {extra_info['discarded_filename']}")
            print(f"Discarded configurations ({extra_info.get('discarded_type', 'above')} threshold): {extra_info['discarded_count']}")
        else:
            print("✗ No discarded configurations to save")
    else:
        # For forces and stress
        print(f"Input file: {input_file}")
        if output_file:
            print(f"Filtered file: {output_file}")
        else:
            print("No output file created (no valid frames)")
        print(f"{property_name.lower()} threshold: {threshold} {units}")
        print(f"Total frames: {total_frames}")
        print(f"Frames kept: {kept_frames} ({kept_frames/total_frames:.4%})" if total_frames > 0 else "Frames kept: 0")
        print(f"Frames discarded: {total_frames - kept_frames}")
    
    print("="*50)


def filter_property(input_file, threshold, property_config):
    """
    Generalized function to filter by any property.
    
    Args:
        input_file (str): Input file path
        threshold (float): Property threshold
        property_config (dict): Configuration for the property to filter
    
    Returns:
        str: Path to the generated output file
    """
    base_name, ext = os.path.splitext(input_file)
    output_file = f"{base_name}{property_config['suffix']}"
    
    # Initialize variables
    print(f"Reading data from {input_file}...")
    data = read(input_file, index=':', format='extxyz')
    total_frames = len(data)
    print(f"Processing {total_frames} frames...")
    
    # Separate configurations based on filtering type
    below_threshold = []
    above_threshold = []
    
    for i, atoms in enumerate(tqdm(data, desc=f"Filtering by {property_config['display_name']}", unit="frame")):
        try:
            # Get property value
            if 'norm_calc' in property_config:
                # For properties requiring norm calculation (forces, stress)
                property_value = property_config['getter'](atoms)
                value = property_config['norm_calc'](property_value)
                
                if value <= threshold:
                    below_threshold.append(atoms)
                else:
                    above_threshold.append(atoms)
            else:
                # For energy (per atom or total)
                property_value = property_config['getter'](atoms)
                
                if property_value < threshold:
                    below_threshold.append(atoms)
                else:
                    above_threshold.append(atoms)
        except (AttributeError, KeyError) as e:
            # Decide where to put the frame if attribute is missing
            if property_config.get('on_error_group', 'discard') == 'above':
                above_threshold.append(atoms)
            else:
                # By default, discard
                pass
    
    # Determine sets to save based on filtering strategy
    if property_config['strategy'] == 'below_threshold':
        main_set = below_threshold
        discarded_set = above_threshold
        group_type = "Below"
        discarded_type = "Above"
    elif property_config['strategy'] == 'largest_group':
        if len(below_threshold) >= len(above_threshold):
            main_set = below_threshold
            discarded_set = above_threshold
            group_type = "Below"
            discarded_type = "Above"
        else:
            main_set = above_threshold
            discarded_set = below_threshold
            group_type = "Above"
            discarded_type = "Below"
    else:
        raise ValueError(f"Unknown filtering strategy: {property_config['strategy']}")
    
    # Save main set
    if main_set:
        print(f"Saving {len(main_set)} configurations to {output_file}...")
        write(output_file, main_set)
    else:
        output_file = None
    
    # Save discarded set if needed
    discarded_filename = None
    if discarded_set and property_config.get('save_discarded', True):
        discarded_filename = f"{base_name}-{property_config['discard_suffix']}"
        print(f"Saving {len(discarded_set)} discarded configurations to {discarded_filename}...")
        write(discarded_filename, discarded_set)
    
    # Prepare extra info for statistics
    extra_info = {
        'below': len(below_threshold),
        'above': len(above_threshold),
        'group_type': group_type,
        'discarded_filename': discarded_filename,
        'discarded_type': discarded_type,
        'discarded_count': len(discarded_set) if discarded_set else 0
    }
    
    # Show statistics
    print_statistics(
        property_config['display_name'],
        input_file,
        output_file,
        threshold,
        total_frames,
        len(main_set),
        property_config['units'],
        extra_info
    )
    
    return output_file


def filter_by_property(input_file, threshold, property_type, save_discarded=False):
    """
    Unified function to filter by any property type.
    
    Args:
        input_file (str): Input file path
        threshold (float): Property threshold
        property_type (str): Type of property ('energy', 'totalenergy', 'forces', 'stress')
        save_discarded (bool): Whether to save discarded frames (default: False)
    
    Returns:
        str: Path to the generated output file
    """
    # Dictionary with configurations for all property types
    property_configs = {
        'eatom': {
            'suffix': '-eatom-filtered.xyz',
            'discard_suffix': 'eatom_discarded.xyz',
            'getter': lambda atoms: atoms.get_potential_energy() / len(atoms),
            'units': 'eV/atom',
            'name': 'energy per atom',
            'display_name': 'energy per atom',
            'strategy': 'largest_group',
            'verbose': False,
            'save_discarded': False,
            'on_error_group': 'discard'
        },
        'toten': {
            'suffix': '-toten-filtered.xyz',
            'discard_suffix': 'toten_discarded.xyz',
            'getter': lambda atoms: atoms.get_potential_energy(),
            'units': 'eV',
            'name': 'total energy',
            'display_name': 'total energy',
            'strategy': 'largest_group',
            'verbose': False,
            'save_discarded': False,
            'on_error_group': 'discard'
        },
        'forces': {
            'suffix': '-forces-filtered.xyz',
            'discard_suffix': 'forces_discarded.xyz',
            'getter': lambda atoms: atoms.get_forces(),
            'norm_calc': lambda x: np.linalg.norm(x),
            'units': 'eV/Å',
            'name': 'force',
            'display_name': 'forces',
            'strategy': 'below_threshold',
            'verbose': True,
            'save_discarded': False,
            'on_error_group': 'discard'
        },
        'stress': {
            'suffix': '-stress-filtered.xyz',
            'discard_suffix': 'stress_discarded.xyz',
            'getter': lambda atoms: atoms.get_stress(),
            'norm_calc': lambda x: np.linalg.norm(x),
            'units': 'eV/Å³',
            'name': 'stress',
            'display_name': 'stress',
            'strategy': 'below_threshold',
            'verbose': True,
            'save_discarded': False,
            'on_error_group': 'discard'
        }
    }
    
    if property_type not in property_configs:
        raise ValueError(f"Unknown property type: {property_type}")
    
    # Apply save_discarded parameter
    config = property_configs[property_type].copy()  # Create a copy to avoid modifying the original
    config['save_discarded'] = save_discarded
    
    return filter_property(input_file, threshold, config)


def show_help():
    """Show help message."""
    print("="*60)
    print("              PROPERTIES FILTER FOR XYZ FILES")
    print("="*60)
    print("DESCRIPTION:")
    print("  Filters atomic configurations by energy, forces, or stress.")
    print()
    print("USAGE:")
    print("  python filter_properties.py <property_type> <threshold> <file.xyz> [--save-discarded]")
    print()
    print("PROPERTY TYPES:")
    print("  eatom        - Filter by energy per atom (eV/atom)")
    print("  toten        - Filter by total energy (eV)")
    print("  forces       - Filter by total force norm (eV/Å)")
    print("  stress       - Filter by total stress norm (eV/Å³)")
    print()
    print("OPTIONS:")
    print("  --save-discarded, -sd  Save discarded frames to a separate file")
    print()
    print("EXAMPLES:")
    print("  python filter_properties.py eatom -2.5 configurations.xyz")
    print("  python filter_properties.py forces 100 trajectory.extxyz --save-discarded")
    print()
    print("OUTPUTS:")
    print("  Filtered configurations are saved to a new file based on the property and threshold.")
    print("  Discarded frames are saved only when --save-discarded option is used.")
    print()
    print("NOTES:")
    print("  • Energy filtering saves the largest group (above/below threshold)")
    print("  • Forces/stress filtering keeps frames below threshold")
    print("  • Compatible with XYZ files from ASE, VASP, CP2K, etc.")
    print("="*60)


def parse_arguments():
    """Parse and validate command line arguments."""
    if len(sys.argv) < 4:
        return None, None, None, False
        
    property_type = sys.argv[1].lower()
    
    try:
        threshold = float(sys.argv[2])
    except ValueError:
        print("Error: Threshold must be a number")
        sys.exit(1)
    
    input_file = sys.argv[3]
    
    if not os.path.isfile(input_file):
        print(f"Error: File {input_file} does not exist")
        sys.exit(1)
    
    # Check for save_discarded flag
    save_discarded = False
    if len(sys.argv) > 4 and sys.argv[4].lower() in ["--save-discarded", "-sd"]:
        save_discarded = True
    
    return property_type, threshold, input_file, save_discarded


def main():
    # Check arguments and show help
    if len(sys.argv) < 4 or sys.argv[1] in ['--help', '-h', '--help']:
        show_help()
        sys.exit(1)
    
    # Parse and validate arguments
    property_type, threshold, input_file, save_discarded = parse_arguments()
    
    # Property configuration and units
    property_types = {
        "eatom": {"units": "eV/atom"},
        "toten": {"units": "eV"},
        "forces": {"units": "eV/Å"},
        "stress": {"units": "eV/Å³"}
    }
    
    # Execute filter
    if property_type not in property_types:
        print(f"Error: Property type '{property_type}' not recognized.")
        print("Valid types: eatom, toten, forces, stress")
        sys.exit(1)
    
    output_file = filter_by_property(input_file, threshold, property_type, save_discarded)
    
    # Final message
    if not output_file:
        print(f"\n❌ Could not create output file.")


if __name__ == "__main__":
    main()
