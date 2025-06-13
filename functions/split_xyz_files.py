#!/usr/bin/env python3
import argparse
import random

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Split an XYZ file into training, validation, and test sets based on percentages.")
parser.add_argument("input_file", type=str, help="Path to the input XYZ file.")
parser.add_argument("--train_ratio", "-tr", type=float, default=0.72, help="Percentage of data for training (default: 0.8).")
parser.add_argument("--validation_ratio", "-vr", type=float, default=0.18, help="Percentage of data for validation (default: 0.1).")
parser.add_argument("--test_ratio", "-ter", type=float, default=0.10, help="Percentage of data for testing (default: 0.1).")
parser.add_argument("--train_output", type=str, default="train.xyz", help="Path to the output training XYZ file (default: train.xyz).")
parser.add_argument("--validation_output", type=str, default="validation.xyz", help="Path to the output validation XYZ file (default: validation.xyz).")
parser.add_argument("--test_output", type=str, default="test.xyz", help="Path to the output test XYZ file (default: test.xyz).")
parser.add_argument("--shuffle", action="store_true", help="Shuffle the configurations before splitting (default: False).")
args = parser.parse_args()

# Validate that the sum of ratios equals 1
if not (0.99 <= args.train_ratio + args.validation_ratio + args.test_ratio <= 1.01):
    raise ValueError("The sum of train_ratio, validation_ratio, and test_ratio must equal 1.")

# Open the input and output files
with open(args.input_file, 'r') as input_file, \
     open(args.train_output, 'w') as train_file, \
     open(args.validation_output, 'w') as validation_file, \
     open(args.test_output, 'w') as test_file:

    config_index = 0  # Initialize configuration index
    configurations = []

    while True:
        # Read the number of atoms and the configuration header
        num_atoms_line = input_file.readline()
        if not num_atoms_line:
            break  # End of file

        header_line = input_file.readline()

        # Read the configuration block
        num_atoms = int(num_atoms_line.strip())
        configuration = [num_atoms_line, header_line] + [input_file.readline() for _ in range(num_atoms)]
        configurations.append(configuration)

    # Shuffle configurations for randomness if shuffle is enabled
    if args.shuffle:
        random.shuffle(configurations)

    # Split configurations into training, validation, and test sets
    num_total = len(configurations)
    num_train = int(num_total * args.train_ratio)
    num_validation = int(num_total * args.validation_ratio)
    num_test = num_total - num_train - num_validation

    train_set = configurations[:num_train]
    validation_set = configurations[num_train:num_train + num_validation]
    test_set = configurations[num_train + num_validation:]

    # Write to respective files
    for config in train_set:
        train_file.writelines(config)

    for config in validation_set:
        validation_file.writelines(config)

    for config in test_set:
        test_file.writelines(config)