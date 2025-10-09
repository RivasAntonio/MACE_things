#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified script for generating datasets from VASP files with different sampling strategies.
Combines the functionality of generate_first_last_files.py, generate_interval_files.py, and generate_train_files.py
"""
import argparse
import os
import random
import subprocess

from ase.io import write
from ase.io import read as ase_read

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback progress function
    def tqdm(iterable, desc=None, total=None):
        return iterable


class AtomsAdaptor(object):
    @classmethod
    def from_file(cls, filename: str, format: str = None):
        """
        Get Atoms from file.
        filename (str): file name which contains structures.
        format (str, optional): file format. If None, will automatically guess.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        if format:
            atoms_list = ase_read(filename, index=":", format=format)
        else:
            try:
                atoms_list = ase_read(filename, index=":")
            except Exception as e:
                raise ValueError(f"Cannot automatically guess the file format: {e}")
        return atoms_list


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
        config_iterator = tqdm(configurations, desc=f"Saving {os.path.basename(output_file)}", disable=not HAS_TQDM) if len(configurations) > 1000 else configurations
        for atoms in config_iterator:
            write(fout, atoms, format="extxyz")


def split_train_test(configurations, test_ratio=0.1, seed=123):
    """Split configurations into train and test sets."""
    random.seed(seed)
    num_test = max(1, int(len(configurations) * test_ratio))
    
    configurations_copy = configurations.copy()
    random.shuffle(configurations_copy)
    test_configs = configurations_copy[:num_test]
    train_configs = configurations_copy[num_test:]
    
    return train_configs, test_configs


def process_first_last_mode(vasp_files, args):
    """Process files to extract first and last N configurations."""
    all_configurations = []
    
    file_iterator = tqdm(vasp_files, desc="Processing files", disable=not HAS_TQDM) if len(vasp_files) > 1 else vasp_files
    for vasp_file in file_iterator:
        try:
            atoms_list = AtomsAdaptor.from_file(filename=vasp_file)
            print(f"File {vasp_file} read successfully. Number of configurations: {len(atoms_list)}")
            
            if len(atoms_list) <= 2 * args.num_configs:
                selected_from_file = atoms_list
                print(f"  File has {len(atoms_list)} configurations <= 2*{args.num_configs}, using all configurations from this file.")
            else:
                first_configs = atoms_list[:args.num_configs]
                last_configs = atoms_list[-args.num_configs:]
                selected_from_file = first_configs + last_configs
                print(f"  Selected first {args.num_configs} and last {args.num_configs} configurations from this file.")
            
            print(f"  Configurations selected from this file: {len(selected_from_file)}")
            all_configurations.extend(selected_from_file)
            
        except Exception as e:
            print(f"Error processing file {vasp_file}: {e}")
    
    return all_configurations


def process_interval_mode(vasp_files, args):
    """Process files to extract configurations at regular intervals."""
    all_configurations = []
    
    file_iterator = tqdm(vasp_files, desc="Processing files", disable=not HAS_TQDM) if len(vasp_files) > 1 else vasp_files
    for vasp_file in file_iterator:
        try:
            print(f"Processing file: {vasp_file}")
            atoms_list = AtomsAdaptor.from_file(filename=vasp_file)
            print(f"File read successfully. Number of configurations: {len(atoms_list)}")
            
            selected_from_file = []
            for i in range(0, len(atoms_list), args.frame_interval):
                selected_from_file.append(atoms_list[i])
            
            print(f"  Selected {len(selected_from_file)} configurations at intervals of {args.frame_interval} from this file.")
            all_configurations.extend(selected_from_file)
            
        except Exception as e:
            print(f"Error processing file {vasp_file}: {e}")
    
    return all_configurations


def process_all_mode(vasp_files, args):
    """Process all configurations from all files."""
    all_configurations = []
    
    file_iterator = tqdm(vasp_files, desc="Processing files", disable=not HAS_TQDM) if len(vasp_files) > 1 else vasp_files
    for vasp_file in file_iterator:
        try:
            atoms_list = AtomsAdaptor.from_file(filename=vasp_file)
            print(f"File {vasp_file} read successfully. Number of configurations: {len(atoms_list)}")
            all_configurations.extend(atoms_list)
        except Exception as e:
            print(f"Error processing file {vasp_file}: {e}")
    
    return all_configurations


def main(args):
    # Find VASP files
    vasp_files = find_vasp_files(args.data_path, args.file_type)
    
    if not vasp_files:
        print(f"No {args.file_type} files found in the path '{args.data_path}'.")
        return
    else:
        print(f"Detected files: {vasp_files}")
    
    # Create directory to save results
    save_dir = args.save_path
    os.makedirs(save_dir, exist_ok=True)
    
    # Process files based on mode
    if args.mode == "first_last":
        all_configurations = process_first_last_mode(vasp_files, args)
        output_file = os.path.join(save_dir, "dataset_first_last.xyz")
        
        if not all_configurations:
            print("No valid configurations found.")
            return
        
        print(f"Total configurations selected from all files: {len(all_configurations)}")
        
        # Save single output file for first_last mode
        save_configurations(all_configurations, output_file)
        if args.relabel:
            relabel_atoms_file(output_file)
            print(f"Dataset file saved as '{output_file}'.")
            print("Labels changed to REF_* in the output file.")
        else:
            print(f"Dataset file saved as '{output_file}'.")
        
    elif args.mode == "interval":
        all_configurations = process_interval_mode(vasp_files, args)
        
        if not all_configurations:
            print("No valid configurations found.")
            return
        
        print(f"Total configurations selected from all files: {len(all_configurations)}")
        
        # Split into train and test
        train_configs, test_configs = split_train_test(all_configurations, seed=args.seed)
        
        print(f"Configurations in the training set: {len(train_configs)}")
        print(f"Configurations in the test set: {len(test_configs)}")
        
        # Save files
        train_file = os.path.join(save_dir, "dataset_interval.xyz")
        test_file = os.path.join(save_dir, "test_interval.xyz")
        
        save_configurations(train_configs, train_file)
        save_configurations(test_configs, test_file)
        
        if args.relabel:
            relabel_atoms_file(train_file)
            relabel_atoms_file(test_file)
            print(f"Training file saved as '{train_file}'.")
            print(f"Test file saved as '{test_file}'.")
            print("Labels changed to REF_* in both output files.")
        else:
            print(f"Training file saved as '{train_file}'.")
            print(f"Test file saved as '{test_file}'.")
        
    elif args.mode == "all":
        all_configurations = process_all_mode(vasp_files, args)
        
        if not all_configurations:
            print("No valid configurations found.")
            return
        
        print(f"Total configurations processed: {len(all_configurations)}")
        
        # Split into train and test
        train_configs, test_configs = split_train_test(all_configurations, seed=args.seed)
        
        print(f"Configurations in the training set: {len(train_configs)}")
        print(f"Configurations in the test set: {len(test_configs)}")
        
        # Save files
        train_file = os.path.join(save_dir, "dataset.xyz")
        test_file = os.path.join(save_dir, "test.xyz")
        
        save_configurations(train_configs, train_file)
        save_configurations(test_configs, test_file)
        
        if args.relabel:
            relabel_atoms_file(train_file)
            relabel_atoms_file(test_file)
            print(f"Dataset file saved as '{train_file}'.")
            print(f"Test file saved as '{test_file}'.")
            print("Labels changed to REF_* in the output files.")
        else:
            print(f"Dataset file saved as '{train_file}'.")
            print(f"Test file saved as '{test_file}'.")


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
  
  # Process all configurations with train/test split
  %(prog)s --mode all --data_path /path/to/vasp/files
        """
    )
    
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["first_last", "interval", "all"],
        required=True,
        help="Processing mode: 'first_last' for first/last N configs, 'interval' for regular intervals, 'all' for all configurations"
    )
    
    parser.add_argument(
        "--data_path",
        "-dp",
        type=str,
        default='.',
        help="Path to the directory containing VASP files."
    )
    
    parser.add_argument(
        "--save_path",
        "-sp",
        type=str,
        default='.',
        help="Path to save the processed data."
    )
    
    parser.add_argument(
        "--file_type",
        "-ft",
        type=str,
        choices=["xml", "OUTCAR", "h5"],
        default="xml",
        help="Type of VASP files to process."
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for randomness in train/test split (used in 'interval' and 'all' modes)."
    )
    
    # Mode-specific arguments
    parser.add_argument(
        "--num_configs",
        "-nc",
        type=int,
        default=1000,
        help="Number of configurations to take from the beginning and end of each file (used in 'first_last' mode)."
    )
    
    parser.add_argument(
        "--frame_interval",
        "-fi",
        type=int,
        default=500,
        help="Interval between frames to extract (used in 'interval' mode)."
    )
    
    parser.add_argument(
        "--relabel",
        "-r",
        action="store_true",
        help="Apply relabeling to change properties to REF_* format (forces->REF_forces, stress->REF_stress, free_energy->REF_energy)."
    )

    args = parser.parse_args()
    main(args)