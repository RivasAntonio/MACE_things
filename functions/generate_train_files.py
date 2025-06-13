#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import random
import subprocess

from ase.io import write
from ase.io import read as ase_read


class AtomsAdaptor(object):
    @classmethod
    def from_file(cls, filename: str, format: str = None):
        """
        Get Atoms from file.
        filename (str): file name which contains structures.
        format (str, optional): file format. If None, will automately guess.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found.")
        if format:
            atoms_list = ase_read(filename, index=":", format=format)
        else:
            try:
                atoms_list = ase_read(filename, index=":")
            except Exception as e:
                raise ValueError(f"Can not automately guess the file format: {e}")
        return atoms_list


def main(args):
    # Initialize list of files
    vasp_files = []
    for root, dirs, files in os.walk(args.data_path):
        for file in files:
            if args.file_type == "xml" and file.endswith("run.xml"):
                vasp_files.append(os.path.join(root, file))
            elif args.file_type == "OUTCAR" and file.endswith("OUTCAR"):
                vasp_files.append(os.path.join(root, file))
            elif args.file_type == "h5" and file.endswith("out.h5"):
                vasp_files.append(os.path.join(root, file))

    # Check if any files were found
    if not vasp_files:
        print(f"No {args.file_type} files found in the path '{args.data_path}'.")
        return
    else:
        print(f"Detected files: {vasp_files}")

    # Split ratios
    train_ratio = args.train_ratio
    validation_ratio = args.validation_ratio

    # Ensure the sum of ratios equals 1
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("The sum of train_ratio and validation_ratio must be less than 1.0 to leave room for the test set.")

    # Create directory to save results
    save_dir = args.save_path
    os.makedirs(save_dir, exist_ok=True)

    all_configurations = []
    atoms_train = []
    atoms_validation = []
    atoms_test = []

    # Set seed for reproducible randomness
    random.seed(args.seed)

    # Process each XML file
    for vasp_file in vasp_files:
        try:
            print(f"Processing file: {vasp_file}")
            # Try to read the file using AtomsAdaptor
            atoms_list = AtomsAdaptor.from_file(filename=vasp_file)
            print(f"File read successfully. Number of configurations: {len(atoms_list)}")
            all_configurations.extend(atoms_list)

        except Exception as e:
            print(f"Error processing file {vasp_file}: {e}")

    num_atoms = len(all_configurations)
    if num_atoms == 0:
        print("No valid configurations found.")
        return
    # Shuffle configurations
    #random.shuffle(all_configurations)
    # Split configurations into training, validation, and test sets
    num_train = int(num_atoms * train_ratio)
    num_validation = int(num_atoms * validation_ratio)
    atoms_train = all_configurations[:num_train]
    atoms_validation = all_configurations[num_train:num_train + num_validation]
    atoms_test = all_configurations[num_train + num_validation:]
    # Save the training, validation, and test sets
    print(f"Training set size: {len(atoms_train)}")
    print(f"Validation set size: {len(atoms_validation)}")
    print(f"Test set size: {len(atoms_test)}")

    # Summary of results
    f = open(os.path.join(save_dir, "summary.txt"), "w")
    f.write(f"Total configurations processed: {num_atoms}\n")
    f.write(f"Configurations in the training set: {len(atoms_train)}\n")
    f.write(f"Configurations in the validation set: {len(atoms_validation)}\n")
    f.write(f"Configurations in the test set: {len(atoms_test)}\n")
    f.close()
    print(f"Total configurations processed: {num_atoms}")
    print(f"Configurations in the training set: {len(atoms_train)}")
    print(f"Configurations in the validation set: {len(atoms_validation)}")
    print(f"Configurations in the test set: {len(atoms_test)}")

    # Save results to files (if necessary)
    if num_atoms > 0:
        train_file = os.path.join(save_dir, "train.extxyz")
        validation_file = os.path.join(save_dir, "validation.extxyz")
        test_file = os.path.join(save_dir, "test.extxyz")

        write(train_file, atoms_train, format="extxyz")
        write(validation_file, atoms_validation, format="extxyz")
        write(test_file, atoms_test, format="extxyz")

        print(f"Sets saved in '{save_dir}'.")

        # Cambiar etiquetas en los archivos generados
        def relabel_atoms_file(input_file):
            sed_cmd = [
                'sed', '-i',
                '-e', 's/forces/REF_forces/g',
                '-e', 's/stress/REF_stress/g',
                '-e', 's/ energy/ REF_energy/g',  # Para no cambiar free_energy
                input_file
            ]
            subprocess.run(sed_cmd, check=True)

        for f in [train_file, validation_file, test_file]:
            relabel_atoms_file(f)
        print("Etiquetas cambiadas a REF_energy, REF_stress y REF_forces en los archivos de salida.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process VASP XML files and split data.")
    parser.add_argument(
        "--data_path",
        "-dp",
        type=str,
        required=True,
        help="Path to the directory containing VASP XML files.",
    )
    parser.add_argument(
        "--save_path",
        "-sp",
        type=str,
        required=True,
        help="Path to save the processed data.",
    )
    parser.add_argument(
        "--train_ratio",
        "-tr",
        type=float,
        default=0.8,
        help="Percentage of data for training.",
    )
    parser.add_argument(
        "--validation_ratio",
        "-vr",
        type=float,
        default=0.1,
        help="Percentage of data for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed for randomness.",
    )
    parser.add_argument(
        "--file_type",
        "-ft",
        type=str,
        choices=["xml", "OUTCAR", "h5"],
        default="xml",
        help="Type of VASP files to process.",
    )

    args = parser.parse_args()
    main(args)

