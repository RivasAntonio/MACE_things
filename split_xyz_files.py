import sys

# Check for correct number of arguments
if len(sys.argv) != 3:
    print("Usage: python split_xyz_files.py <input_file.xyz> <indices_file.txt>")
    sys.exit(1)

# Get file paths from command-line arguments
input_xyz_file = sys.argv[1]
indices_file = sys.argv[2]

# Read the indices for validation
with open(indices_file, 'r') as f:
    validation_indices = set(map(int, f.readlines()))

# Open the input and output files
with open(input_xyz_file, 'r') as input_file, \
     open('validation.xyz', 'w') as validation_file, \
     open('train.xyz', 'w') as train_file:

    config_index = 0  # Initialize configuration index

    while True:
        # Read the number of atoms and the configuration header
        num_atoms_line = input_file.readline()
        if not num_atoms_line:
            break  # End of file

        header_line = input_file.readline()

        # Read the configuration block
        num_atoms = int(num_atoms_line.strip())
        configuration = [input_file.readline() for _ in range(num_atoms)]

        # Use the current configuration index
        if config_index in validation_indices:
            validation_file.write(num_atoms_line)
            validation_file.write(header_line)
            validation_file.writelines(configuration)
        else:
            train_file.write(num_atoms_line)
            train_file.write(header_line)
            train_file.writelines(configuration)

        config_index += 1  # Increment configuration index