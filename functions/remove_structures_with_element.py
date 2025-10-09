#!/usr/bin/env python3
"""
remove_by_element.py

Elimina todas las configuraciones de un EXTXYZ que contengan
un elemento especificado por el usuario.

Uso:
    python remove_by_element.py archivo.extxyz Elemento

Ejemplo:
    python remove_by_element.py full.extxyz Cl
"""

import sys
from ase.io import read, write

def remove_by_element(xyz_file, element):
    atoms_list = read(xyz_file, index=":", format="extxyz")
    total_structures = len(atoms_list)
    filtered = [atoms for atoms in atoms_list if element not in atoms.get_chemical_symbols()]
    kept_count = len(filtered)
    removed_count = total_structures - kept_count
    return filtered, total_structures, kept_count, removed_count

def main():
    if len(sys.argv) < 3:
        print("Uso: python remove_by_element.py archivo.extxyz Elemento")
        sys.exit(1)
    
    input_file = sys.argv[1]
    element = sys.argv[2]
    output_file = input_file.replace(".extxyz", f"_without_{element}.extxyz")
    
    filtered_atoms, total_structures, kept_count, removed_count = remove_by_element(input_file, element)
    
    print(f"Estructuras totales en el archivo original: {total_structures}")
    print(f"Estructuras con el elemento {element} (eliminadas): {removed_count}")
    print(f"Estructuras sin el elemento {element} (conservadas): {kept_count}")
    
    if filtered_atoms:
        write(output_file, filtered_atoms)
        print(f"Se guardaron {kept_count} configuraciones en: {output_file}")
    else:
        print(f"Todas las configuraciones contenían {element}, archivo vacío.")

if __name__ == "__main__":
    main()
