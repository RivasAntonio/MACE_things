#!/usr/bin/env python3
"""Split an XYZ file into train/validation/test sets while preserving origin groups."""

import argparse
import random
from ase.io import read, write
from collections import defaultdict

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Split an XYZ file into training, validation, and test sets based on percentages, keeping structures from the same origin together.")
parser.add_argument("input_file", type=str, help="Path to the input XYZ file.")
parser.add_argument("--train_ratio", "-tr", type=float, default=0.8, help="Percentage of data for training (default: 0.8).")
parser.add_argument("--validation_ratio", "-vr", type=float, default=0.2, help="Percentage of data for validation (default: 0.1).")
parser.add_argument("--test_ratio", "-ter", type=float, default=0.0, help="Percentage of data for testing (default: 0.1).")
parser.add_argument("--train_output", type=str, default="train.xyz", help="Path to the output training XYZ file (default: train.xyz).")
parser.add_argument("--validation_output", type=str, default="validation.xyz", help="Path to the output validation XYZ file (default: validation.xyz).")
parser.add_argument("--test_output", type=str, default="test.xyz", help="Path to the output test XYZ file (default: test.xyz).")
parser.add_argument("--no-shuffle", action="store_true", help="Do NOT shuffle the origin groups before splitting (default: shuffle enabled).")
args = parser.parse_args()


# Validate that the sum of ratios equals 1
if not (0.99 <= args.train_ratio + args.validation_ratio + args.test_ratio <= 1.01):
    raise ValueError("The sum of train_ratio, validation_ratio, and test_ratio must equal 1.")

# Read all configurations from the XYZ file
print(f"Reading configurations from {args.input_file}...")
configurations = read(args.input_file, index=':')
print(f"Total configurations read: {len(configurations)}")

# Group configurations by origin
origin_groups = defaultdict(list)
for atoms in configurations:
    origin = atoms.info.get('origin', 'unknown')
    origin_groups[origin].append(atoms)

print(f"Found {len(origin_groups)} different origins:")
for origin, atoms_list in origin_groups.items():
    print(f"  - {origin}: {len(atoms_list)} structures")

# Convert to a list of (origin, atoms_list) tuples
origin_list = list(origin_groups.items())

# Shuffle origin groups by default (unless --no-shuffle is specified)
if not args.no_shuffle:
    print("Shuffling origin groups...")
    random.shuffle(origin_list)
else:
    print("Skipping shuffle (keeping original order)...")

# Split origin groups into train, validation, and test sets
# We'll distribute complete origin groups to maintain data integrity
train_set = []
validation_set = []
test_set = []

# Calculate target counts
num_total = len(configurations)
target_train = int(num_total * args.train_ratio)
target_validation = int(num_total * args.validation_ratio)

current_train = 0
current_validation = 0
current_test = 0

for origin, atoms_list in origin_list:
    # Decide which set to add this origin group to
    # Try to get as close as possible to the target ratios
    if current_train < target_train:
        train_set.extend(atoms_list)
        current_train += len(atoms_list)
    elif current_validation < target_validation:
        validation_set.extend(atoms_list)
        current_validation += len(atoms_list)
    else:
        test_set.extend(atoms_list)
        current_test += len(atoms_list)

# Print statistics
print(f"\nSplit summary:")
print(f"  Training set: {len(train_set)} structures ({len(train_set)/num_total*100:.2f}%)")
print(f"  Validation set: {len(validation_set)} structures ({len(validation_set)/num_total*100:.2f}%)")
print(f"  Test set: {len(test_set)} structures ({len(test_set)/num_total*100:.2f}%)")

# Write to respective files
print(f"\nWriting training set to {args.train_output}...")
write(args.train_output, train_set, format='extxyz')

print(f"Writing validation set to {args.validation_output}...")
write(args.validation_output, validation_set, format='extxyz')

print(f"Writing test set to {args.test_output}...")
write(args.test_output, test_set, format='extxyz')

print("\nDone!")
