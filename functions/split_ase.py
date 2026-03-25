"""Split an XYZ dataset into train, validation, and test subsets."""

import random
from ase.io import read, write

# Cargar todas las configuraciones
atoms_list = read('all.xyz', index=':')

# Mezclar aleatoriamente para asegurar una distribución uniforme
random.seed(42) # Para reproducibilidad
random.shuffle(atoms_list)

n_total = len(atoms_list)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)

train_data = atoms_list[:n_train]
val_data = atoms_list[n_train : n_train + n_val]
test_data = atoms_list[n_train + n_val:]

write('train.xyz', train_data)
write('val.xyz', val_data)
write('test.xyz', test_data)

print(f"Dividido en: Train ({len(train_data)}), Val ({len(val_data)}), Test ({len(test_data)})")
