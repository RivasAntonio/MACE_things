#!/usr/bin/env python3
import re
import sys
import os.path
from tqdm import tqdm

def convert_stress(xyz_content, conversion_factor=-0.0006241509):
    """
    Convert stress values from kbar to meV/Å³ in an XYZ file for all configurations.
    
    Args:
        xyz_content (str): Content of the XYZ file
        conversion_factor (float): Conversion factor from kbar to meV/Å³
    
    Returns:
        str: Updated XYZ file content with converted stress values
    """
    lines = xyz_content.strip().split('\n')
    line_index = 0
    output_lines = []
    
    while line_index < len(lines):
        # Get the number of atoms in this configuration
        if not lines[line_index].strip().isdigit():
            # Skip non-numeric lines (shouldn't happen in a well-formed XYZ file)
            output_lines.append(lines[line_index])
            line_index += 1
            continue
            
        n_atoms = int(lines[line_index])
        output_lines.append(lines[line_index])  # Add atom count to output
        line_index += 1
        
        # Process the header line if we haven't reached the end of the file
        if line_index < len(lines):
            header_line = lines[line_index]
            # Find stress in the header line
            stress_match = re.search(r'stress="([^"]+)"', header_line)
            
            if stress_match:
                stress_values_str = stress_match.group(1)
                stress_values = [float(val) for val in stress_values_str.split()]
                
                # Convert stress values
                converted_stress = [val * conversion_factor for val in stress_values]
                
                # Format converted stress values consistently
                converted_stress_str = " ".join([f"{val:.16f}" for val in converted_stress])
                
                # Replace stress values in header
                new_header_line = re.sub(r'stress="[^"]+"', f'stress="{converted_stress_str}"', header_line)
                output_lines.append(new_header_line)
            else:
                # No stress in this header, keep it as is
                output_lines.append(header_line)
                
            line_index += 1
        
        # Add the atom coordinate lines for this configuration
        for _ in range(n_atoms):
            if line_index < len(lines):
                output_lines.append(lines[line_index])
                line_index += 1
            else:
                break
    
    return '\n'.join(output_lines)

def main():
    # Check if the correct number of arguments were provided
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python convert_units.py input.xyz [conversion_factor]")
        sys.exit(1)
    
    # Get the input filename from command line arguments
    input_file = sys.argv[1]
    
    # Get conversion factor if provided, else use default
    if len(sys.argv) == 3:
        try:
            conversion_factor = float(sys.argv[2])
        except ValueError:
            print("Error: conversion_factor must be a float.")
            sys.exit(1)
    else:
        conversion_factor = -0.0006241509
    
    # Generate output filename based on input filename
    input_basename = os.path.basename(input_file)
    input_name, input_ext = os.path.splitext(input_basename)
    output_file = f"{input_name}_converted{input_ext}"
    
    try:
        # Read the input file
        with open(input_file, 'r') as f:
            xyz_content = f.read()
        
        # Count how many configurations were processed
        atom_count_lines = [line for line in xyz_content.split('\n') if line.strip().isdigit()]
        
        # Convert the stress values with a progress bar
        with tqdm(total=len(atom_count_lines), desc="Converting stress values") as pbar:
            converted_content = convert_stress(xyz_content, conversion_factor=conversion_factor)
            pbar.update(len(atom_count_lines))
        
        # Write the output to a new file
        with open(output_file, 'w') as f:
            f.write(converted_content)
        
        print(f"Conversion complete. Output written to {output_file}")
        
        print(f"Processed {len(atom_count_lines)} configurations in the XYZ file")
        
    
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()