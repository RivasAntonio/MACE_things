#!/usr/bin/env python3
"""
filter_by_element.py

Guarda todas las configuraciones de un EXTXYZ que contengan
un elemento especificado por el usuario.

Uso:
    python filter_by_element.py archivo.extxyz Elemento

Ejemplo:
    python filter_by_element.py full.extxyz Cl
"""

import sys
from ase.io import read, write

def filter_by_element(xyz_file, element):
    atoms_list = read(xyz_file, index=":", format="extxyz")
    total_structures = len(atoms_list)
    filtered = [atoms for atoms in atoms_list if element in atoms.get_chemical_symbols()]
    filtered_count = len(filtered)
    removed_count = total_structures - filtered_count
    return filtered, total_structures, filtered_count, removed_count

def main():
    if len(sys.argv) < 3:
        print("Uso: python filter_by_element.py archivo.extxyz Elemento")
        sys.exit(1)
    
    input_file = sys.argv[1]
    element = sys.argv[2]
    output_file = input_file.replace(".extxyz", f"_with_{element}.extxyz")
    
    filtered_atoms, total_structures, filtered_count, removed_count = filter_by_element(input_file, element)
    
    print(f"Estructuras totales en el archivo original: {total_structures}")
    print(f"Estructuras con el elemento {element}: {filtered_count}")
    print(f"Estructuras sin el elemento {element} (eliminadas): {removed_count}")
    
    if filtered_atoms:
        write(output_file, filtered_atoms)
        print(f"Se guardaron {filtered_count} configuraciones en: {output_file}")
    else:
        print(f"No se encontró ninguna configuración con el elemento {element}.")

if __name__ == "__main__":
    main()
