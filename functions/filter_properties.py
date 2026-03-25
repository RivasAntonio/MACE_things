#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to filter atomic configurations based on different properties.
Can filter by energy per atom, total energy, forces, stress, and pressure.
"""
import sys
from ase.io import read, write
import numpy as np
import os
from tqdm import tqdm

# Conversion constant: 1 eV/Å³ = 160.21766208 GPa
EV_PER_A3_TO_GPA = 160.21766208


def write_filtered_frames(output_file, valid_frames):
    """
    Helper function to write filtered frames to a file efficiently.
    
    Args:
        output_file (str): Path to output file
        valid_frames (list): List of valid frames
    """
    if not valid_frames:
        return False
    
    # Write all frames at once - much faster than appending
    write(output_file, valid_frames, format='extxyz')
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
        write(output_file, main_set, format='extxyz')
    else:
        output_file = None
    
    # Save discarded set if needed
    discarded_filename = None
    if discarded_set and property_config.get('save_discarded', True):
        discarded_filename = f"{base_name}-{property_config['discard_suffix']}"
        print(f"Saving {len(discarded_set)} discarded configurations to {discarded_filename}...")
        write(discarded_filename, discarded_set, format='extxyz')
    
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


def filter_by_energy_range(input_file, energy_range, save_discarded=True):
    """
    Filter configurations by energy per atom range, discarding configurations within the range.
    
    Args:
        input_file (str): Input file path
        energy_range (tuple): Energy range (min, max) in eV/atom to discard
        save_discarded (bool): Whether to save discarded frames (default: True)
    
    Returns:
        str: Path to the generated output file
    """
    base_name, ext = os.path.splitext(input_file)
    output_file = f"{base_name}-eatom-range-filtered.xyz"
    
    # Parse energy range
    emin, emax = energy_range
    
    # Initialize variables
    print(f"Reading data from {input_file}...")
    data = read(input_file, index=':', format='extxyz')
    total_frames = len(data)
    print(f"Processing {total_frames} frames...")
    
    # Separate configurations
    inside_range = []
    outside_range = []
    
    for i, atoms in enumerate(tqdm(data, desc=f"Filtering by energy range [{emin}, {emax}] eV/atom", unit="frame")):
        try:
            energy_per_atom = atoms.get_potential_energy() / len(atoms)
            
            if emin <= energy_per_atom <= emax:
                inside_range.append(atoms)
            else:
                outside_range.append(atoms)
        except (AttributeError, KeyError) as e:
            # If energy not available, discard
            pass
    
    # Save configurations outside the range (kept)
    if outside_range:
        print(f"Saving {len(outside_range)} configurations (outside range) to {output_file}...")
        write(output_file, outside_range, format='extxyz')
    else:
        output_file = None
    
    # Save discarded set (inside range) if needed
    discarded_filename = None
    if inside_range and save_discarded:
        discarded_filename = f"{base_name}-eatom_range_discarded.xyz"
        print(f"Saving {len(inside_range)} discarded configurations (inside range) to {discarded_filename}...")
        write(discarded_filename, inside_range, format='extxyz')
    
    # Show statistics
    print(f"\n{'='*50}")
    print(f"ENERGY RANGE FILTERING STATISTICS")
    print("="*50)
    print(f"Energy range: [{emin}, {emax}] eV/atom")
    print(f"Total configurations read: {total_frames}")
    print(f"Configurations inside range (discarded): {len(inside_range)}")
    print(f"Configurations outside range (kept): {len(outside_range)}")
    print(f"✓ Main file saved: {output_file if output_file else 'None'}")
    print(f"Total configurations saved: {len(outside_range)}")
    if discarded_filename:
        print(f"✓ Discarded file saved: {discarded_filename}")
        print(f"Discarded configurations (inside range): {len(inside_range)}")
    elif inside_range:
        print("✗ Discarded configurations not saved (default: save; use --no-save-discarded to disable saving)")
    else:
        print("✗ No discarded configurations")
    print("="*50)
    
    return output_file


def filter_by_property(input_file, threshold, property_type, save_discarded=True):
    """
    Unified function to filter by any property type.
    
    Args:
        input_file (str): Input file path
        threshold (float): Property threshold
        property_type (str): Type of property ('energy', 'totalenergy', 'forces', 'stress')
        save_discarded (bool): Whether to save discarded frames (default: True)
    
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
            'save_discarded': True,
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
            'save_discarded': True,
            'on_error_group': 'discard'
        },
        'forces': {
            'suffix': '-forces-filtered.xyz',
            'discard_suffix': 'forces_discarded.xyz',
            'getter': lambda atoms: atoms.get_forces(),
            'norm_calc': lambda x: np.linalg.norm(x, axis=1).max(),
            'units': 'eV/Å',
            'name': 'force',
            'display_name': 'forces',
            'strategy': 'below_threshold',
            'verbose': True,
            'save_discarded': True,
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
            'save_discarded': True,
            'on_error_group': 'discard'
        },
        'pressure': {
            'suffix': '-pressure-filtered.xyz',
            'discard_suffix': 'pressure_discarded.xyz',
            'getter': lambda atoms: atoms.get_stress(),
            'norm_calc': lambda x: (x[0] + x[1] + x[2]) / 3.0 * EV_PER_A3_TO_GPA,
            'units': 'GPa',
            'name': 'pressure',
            'display_name': 'pressure',
            'strategy': 'below_threshold',
            'verbose': True,
            'save_discarded': True,
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
    print("  Single filter:")
    print("    python filter_properties.py <property_type> <threshold> <file1.xyz> [file2.xyz ...] [options]")
    print("    python filter_properties.py erange <min,max> <file1.xyz> [file2.xyz ...] [options]")
    print()
    print("  Multiple filters (sequential):")
    print("    python filter_properties.py --multi <file1.xyz> [file2.xyz ...] [options]")
    print("        --filter <property_type> <threshold>")
    print("        [--filter <property_type> <threshold> ...]")
    print()
    print("PROPERTY TYPES:")
    print("  eatom        - Filter by energy per atom (eV/atom)")
    print("  toten        - Filter by total energy (eV)")
    print("  forces       - Filter by total force norm (eV/Å)")
    print("  stress       - Filter by total stress norm (eV/Å³)")
    print("  pressure     - Filter by pressure (1/3 trace of stress tensor, GPa)")
    print("  erange       - Discard configurations within energy/atom range (eV/atom)")
    print()
    print("OPTIONS:")
    print("  --no-save-discarded, -nsd  Do NOT save discarded frames (default: save)")
    print("  --save-discarded, -sd      Explicitly request saving discarded frames (default behaviour)")
    print("  --multi                    Enable multiple sequential filters")
    print("  --filter <type> <thresh>   Add a filter (use with --multi)")
    print()
    print("EXAMPLES:")
    print("  Single filter:")
    print("    python filter_properties.py eatom -2.5 configurations.xyz")
    print("    python filter_properties.py forces 100 traj1.extxyz traj2.extxyz")
    print("    python filter_properties.py forces 100 traj1.extxyz traj2.extxyz -nsd")
    print("    python filter_properties.py erange -10,-9 data1.xyz data2.xyz")
    print()
    print("  Multiple filters:")
    print("    python filter_properties.py --multi data.xyz \\")
    print("        --filter forces 50 --filter stress 10 --filter eatom -3.0")
    print("    python filter_properties.py --multi data.xyz \\")
    print("        --filter erange -10,-9 --filter forces 100 -nsd")
    print("    python filter_properties.py pressure 50 data.xyz")
    print()
    print("OUTPUTS:")
    print("  Single filter: Output file with suffix based on property type")
    print("  Multiple filters: Output file with '-multi-filtered.xyz' suffix")
    print("  Discarded frames are saved by default; use --no-save-discarded to disable")
    print("  When multiple files are provided, each is processed independently")
    print()
    print("NOTES:")
    print("  • Energy filtering saves the largest group (above/below threshold)")
    print("  • Forces/stress filtering keeps frames below threshold")
    print("  • Energy range filtering discards configurations within the range")
    print("  • Multiple filters are applied sequentially in the order specified")
    print("  • Compatible with XYZ files from ASE, VASP, CP2K, etc.")
    print("="*60)


def parse_arguments():
    """Parse and validate command line arguments."""
    if len(sys.argv) < 4:
        return None, None, None, False, None
    
    # Check if multi-filter mode
    if sys.argv[1].lower() == '--multi':
        return parse_multi_filter_arguments()
    
    # Single filter mode
    property_type = sys.argv[1].lower()
    
    # Special handling for energy range
    if property_type == 'erange':
        try:
            range_str = sys.argv[2]
            range_parts = range_str.split(',')
            if len(range_parts) != 2:
                print("Error: Energy range must be in format 'min,max' (e.g., '-10,-9')")
                sys.exit(1)
            threshold = (float(range_parts[0]), float(range_parts[1]))
            if threshold[0] >= threshold[1]:
                print("Error: Minimum energy must be less than maximum energy")
                sys.exit(1)
        except ValueError:
            print("Error: Energy range values must be numbers (e.g., '-10,-9')")
            sys.exit(1)
    else:
        try:
            threshold = float(sys.argv[2])
        except ValueError:
            print("Error: Threshold must be a number")
            sys.exit(1)
    
    # Collect all input files (starting from argv[3] until we hit a flag) -- default: save discarded
    input_files = []
    save_discarded = True
    
    for i in range(3, len(sys.argv)):
        arg = sys.argv[i]
        if arg.lower() in ["--no-save-discarded", "-nsd"]:
            # Explicit flag to NOT save discarded configurations
            save_discarded = False
            break
        if arg.lower() in ["--save-discarded", "-sd"]:
            save_discarded = True
            break
        else:
            if not os.path.isfile(arg):
                print(f"Error: File {arg} does not exist")
                sys.exit(1)
            input_files.append(arg)
    
    if not input_files:
        print("Error: No input files specified")
        sys.exit(1)
    
    return property_type, threshold, input_files, save_discarded, None


def parse_multi_filter_arguments():
    """Parse arguments for multiple filter mode."""
    # Format: --multi file.xyz [file2.xyz ...] [--filter type thresh] [--filter type thresh] [options]
    
    input_files = []
    filters = []
    save_discarded = True
    
    i = 2  # Start after '--multi'
    
    # First, collect input files
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.lower() == '--filter':
            break
        elif arg.lower() in ["--no-save-discarded", "-nsd"]:
            save_discarded = False
            i += 1
        elif arg.lower() in ["--save-discarded", "-sd"]:
            save_discarded = True
            i += 1
        else:
            if not os.path.isfile(arg):
                print(f"Error: File {arg} does not exist")
                sys.exit(1)
            input_files.append(arg)
            i += 1
    
    if not input_files:
        print("Error: No input files specified for --multi mode")
        sys.exit(1)
    
    # Now collect filters
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg.lower() == '--filter':
            if i + 2 >= len(sys.argv):
                print("Error: --filter requires <property_type> <threshold>")
                sys.exit(1)
            
            property_type = sys.argv[i + 1].lower()
            threshold_str = sys.argv[i + 2]
            
            # Parse threshold (can be single value or range)
            if property_type == 'erange':
                try:
                    range_parts = threshold_str.split(',')
                    if len(range_parts) != 2:
                        print("Error: Energy range must be in format 'min,max' (e.g., '-10,-9')")
                        sys.exit(1)
                    threshold = (float(range_parts[0]), float(range_parts[1]))
                    if threshold[0] >= threshold[1]:
                        print("Error: Minimum energy must be less than maximum energy")
                        sys.exit(1)
                except ValueError:
                    print("Error: Energy range values must be numbers")
                    sys.exit(1)
            else:
                try:
                    threshold = float(threshold_str)
                except ValueError:
                    print(f"Error: Threshold must be a number, got '{threshold_str}'")
                    sys.exit(1)
            
            filters.append((property_type, threshold))
            i += 3
        elif arg.lower() in ["--no-save-discarded", "-nsd"]:
            save_discarded = False
            i += 1
        elif arg.lower() in ["--save-discarded", "-sd"]:
            save_discarded = True
            i += 1
        else:
            print(f"Error: Unexpected argument '{arg}' in --multi mode")
            sys.exit(1)
    
    if not filters:
        print("Error: No filters specified for --multi mode. Use --filter <type> <threshold>")
        sys.exit(1)
    
    return 'multi', None, input_files, save_discarded, filters


def apply_multiple_filters(input_file, filters, save_discarded):
    """
    Apply multiple filters sequentially to a file.
    
    Args:
        input_file (str): Input file path
        filters (list): List of tuples (property_type, threshold)
        save_discarded (bool): Whether to save discarded frames
    
    Returns:
        str: Path to the final output file
    """
    base_name, ext = os.path.splitext(input_file)
    
    print(f"\n{'='*60}")
    print(f"APPLYING {len(filters)} SEQUENTIAL FILTERS")
    print(f"{'='*60}")
    
    # Show filter plan
    print("\nFilter sequence:")
    for idx, (prop_type, threshold) in enumerate(filters, 1):
        if prop_type == 'erange':
            print(f"  {idx}. Energy range: [{threshold[0]}, {threshold[1]}] eV/atom")
        else:
            units = {"eatom": "eV/atom", "toten": "eV", "forces": "eV/Å", "stress": "eV/Å³", "pressure": "GPa"}
            print(f"  {idx}. {prop_type}: {threshold} {units.get(prop_type, '')}")
    print()
    
    # Read initial data
    print(f"Reading initial data from {input_file}...")
    current_data = read(input_file, index=':', format='extxyz')
    initial_count = len(current_data)
    print(f"Initial configurations: {initial_count}")
    
    # Track all discarded frames
    all_discarded = []
    
    # Apply filters sequentially
    for idx, (prop_type, threshold) in enumerate(filters, 1):
        print(f"\n{'-'*60}")
        print(f"Applying filter {idx}/{len(filters)}: {prop_type}")
        print(f"{'-'*60}")
        
        filtered_data = []
        discarded_data = []
        
        # Property configurations
        property_configs = {
            'eatom': {
                'getter': lambda atoms: atoms.get_potential_energy() / len(atoms),
                'comparison': lambda val, thresh: abs(val) < abs(thresh),  # Largest group logic
            },
            'toten': {
                'getter': lambda atoms: atoms.get_potential_energy(),
                'comparison': lambda val, thresh: abs(val) < abs(thresh),  # Largest group logic
            },
            'forces': {
                'getter': lambda atoms: np.linalg.norm(atoms.get_forces(), axis=1).max(),
                'comparison': lambda val, thresh: val <= thresh,
            },
            'stress': {
                'getter': lambda atoms: np.linalg.norm(atoms.get_stress()),
                'comparison': lambda val, thresh: val <= thresh,
            },
            'pressure': {
                'getter': lambda atoms: (atoms.get_stress()[0] + atoms.get_stress()[1] + atoms.get_stress()[2]) / 3.0 * EV_PER_A3_TO_GPA,
                'comparison': lambda val, thresh: val <= thresh,
            }
        }
        
        if prop_type == 'erange':
            # Energy range filtering
            emin, emax = threshold
            for atoms in tqdm(current_data, desc="Filtering", unit="frame"):
                try:
                    energy_per_atom = atoms.get_potential_energy() / len(atoms)
                    # Keep if OUTSIDE range
                    if not (emin <= energy_per_atom <= emax):
                        filtered_data.append(atoms)
                    else:
                        discarded_data.append(atoms)
                except (AttributeError, KeyError):
                    pass
        elif prop_type in ['eatom', 'toten']:
            # Energy filtering - keep largest group
            below_thresh = []
            above_thresh = []
            
            for atoms in tqdm(current_data, desc="Filtering", unit="frame"):
                try:
                    value = property_configs[prop_type]['getter'](atoms)
                    if value < threshold:
                        below_thresh.append(atoms)
                    else:
                        above_thresh.append(atoms)
                except (AttributeError, KeyError):
                    pass
            
            # Keep largest group
            filtered_data = below_thresh if len(below_thresh) >= len(above_thresh) else above_thresh
            discarded_data = above_thresh if len(below_thresh) >= len(above_thresh) else below_thresh
            kept_type = "below" if len(below_thresh) >= len(above_thresh) else "above"
            print(f"Kept largest group ({kept_type} threshold): {len(filtered_data)} configurations")
        else:
            # Forces/stress/pressure filtering - keep below threshold
            for atoms in tqdm(current_data, desc="Filtering", unit="frame"):
                try:
                    value = property_configs[prop_type]['getter'](atoms)
                    if property_configs[prop_type]['comparison'](value, threshold):
                        filtered_data.append(atoms)
                    else:
                        discarded_data.append(atoms)
                except (AttributeError, KeyError):
                    pass
        
        print(f"Before: {len(current_data)} → After: {len(filtered_data)} configurations")
        print(f"Discarded in this step: {len(discarded_data)} configurations")
        
        # Add discarded to the accumulated list
        all_discarded.extend(discarded_data)
        
        current_data = filtered_data
        
        if not current_data:
            print("\n⚠️  Warning: No configurations remaining after this filter!")
            return None
    
    # Save final result
    output_file = f"{base_name}-multi-filtered.xyz"
    print(f"\n{'='*60}")
    print(f"SAVING FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Writing {len(current_data)} configurations to {output_file}...")
    write(output_file, current_data, format='extxyz')
    
    # Save discarded configurations if requested
    discarded_filename = None
    if all_discarded and save_discarded:
        discarded_filename = f"{base_name}-multi-filtered_discarded.xyz"
        print(f"Writing {len(all_discarded)} discarded configurations to {discarded_filename}...")
        write(discarded_filename, all_discarded, format='extxyz')
    
    # Statistics
    print(f"\n{'='*60}")
    print(f"MULTI-FILTER SUMMARY")
    print(f"{'='*60}")
    print(f"Initial configurations: {initial_count}")
    print(f"Final configurations: {len(current_data)}")
    print(f"Total discarded configurations: {len(all_discarded)}")
    print(f"Reduction: {initial_count - len(current_data)} ({(initial_count - len(current_data))/initial_count:.2%})")
    print(f"✓ Main file saved: {output_file}")
    if discarded_filename:
        print(f"✓ Discarded file saved: {discarded_filename}")
    elif all_discarded:
        print(f"✗ Discarded file not saved (use --save-discarded to enable)")
    else:
        print(f"✗ No configurations were discarded")
    print(f"{'='*60}")
    
    return output_file


def main():
    # Check arguments and show help
    if len(sys.argv) < 4 or sys.argv[1] in ['--help', '-h', '--help']:
        show_help()
        sys.exit(1)
    
    # Parse and validate arguments
    property_type, threshold, input_files, save_discarded, filters = parse_arguments()
    
    # Property configuration and units
    property_types = {
        "eatom": {"units": "eV/atom"},
        "toten": {"units": "eV"},
        "forces": {"units": "eV/Å"},
        "stress": {"units": "eV/Å³"},
        "pressure": {"units": "GPa"},
        "erange": {"units": "eV/atom"},
        "multi": {"units": "various"}
    }
    
    # Check if multi-filter mode
    if property_type == 'multi':
        # Process each input file with multiple filters
        total_files = len(input_files)
        successful_files = 0
        
        for idx, input_file in enumerate(input_files, 1):
            if total_files > 1:
                print(f"\n{'='*60}")
                print(f"Processing file {idx}/{total_files}: {input_file}")
                print(f"{'='*60}")
            
            output_file = apply_multiple_filters(input_file, filters, save_discarded)
            
            if output_file:
                successful_files += 1
            else:
                print(f"\n❌ Could not create output file for {input_file}")
        
        # Final summary for multiple files
        if total_files > 1:
            print(f"\n{'='*60}")
            print(f"SUMMARY: Processed {successful_files}/{total_files} files successfully")
            print(f"{'='*60}")
        
        return
    
    # Execute single filter
    if property_type not in property_types:
        print(f"Error: Property type '{property_type}' not recognized.")
        print("Valid types: eatom, toten, forces, stress, pressure, erange")
        sys.exit(1)
    
    # Process each input file
    total_files = len(input_files)
    successful_files = 0
    
    for idx, input_file in enumerate(input_files, 1):
        if total_files > 1:
            print(f"\n{'='*60}")
            print(f"Processing file {idx}/{total_files}: {input_file}")
            print(f"{'='*60}")
        
        # Handle energy range filtering separately
        if property_type == 'erange':
            output_file = filter_by_energy_range(input_file, threshold, save_discarded)
        else:
            output_file = filter_by_property(input_file, threshold, property_type, save_discarded)
        
        # Track success
        if output_file:
            successful_files += 1
        else:
            print(f"\n❌ Could not create output file for {input_file}")
    
    # Final summary for multiple files
    if total_files > 1:
        print(f"\n{'='*60}")
        print(f"SUMMARY: Processed {successful_files}/{total_files} files successfully")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
