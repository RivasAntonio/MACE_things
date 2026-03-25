#!/usr/bin/env python3
"""
keep_structures_with_element.py

Mantiene solo las configuraciones de un EXTXYZ que contengan
los elementos especificados por el usuario.

Modos de filtrado:
  --any  (por defecto): conserva estructuras que contengan AL MENOS UNO de los elementos.
  --all               : conserva estructuras que contengan TODOS los elementos.

Uso:
    python keep_structures_with_element.py archivo.extxyz Elem1 [Elem2 ...] [--any|--all]

Ejemplos:
    python keep_structures_with_element.py full.extxyz Cl
    python keep_structures_with_element.py full.extxyz Si O --all
    python keep_structures_with_element.py full.extxyz Na Cl --any
"""

import sys
import argparse
from ase.io import read, write

def keep_by_elements(xyz_file, elements, mode="any"):
    atoms_list = read(xyz_file, index=":", format="extxyz")
    total_structures = len(atoms_list)
    elements_set = set(elements)

    if mode == "all":
        filtered = [
            atoms for atoms in atoms_list
            if elements_set.issubset(set(atoms.get_chemical_symbols()))
        ]
    else:  # any
        filtered = [
            atoms for atoms in atoms_list
            if elements_set & set(atoms.get_chemical_symbols())
        ]

    kept_count = len(filtered)
    removed_count = total_structures - kept_count
    return filtered, total_structures, kept_count, removed_count

def main():
    parser = argparse.ArgumentParser(
        description="Filtra configuraciones EXTXYZ según los elementos presentes."
    )
    parser.add_argument("input_file", help="Archivo EXTXYZ de entrada")
    parser.add_argument("elements", nargs="+", help="Elemento(s) a buscar")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--any", dest="mode", action="store_const", const="any",
        help="Conservar estructuras con AL MENOS UNO de los elementos (por defecto)"
    )
    mode_group.add_argument(
        "--all", dest="mode", action="store_const", const="all",
        help="Conservar estructuras que contengan TODOS los elementos"
    )
    parser.set_defaults(mode="any")

    args = parser.parse_args()

    input_file = args.input_file
    elements = args.elements
    mode = args.mode

    elements_str = "_".join(elements)
    output_file = input_file.replace(".extxyz", f"_with_{elements_str}_{mode}.extxyz")

    filtered_atoms, total_structures, kept_count, removed_count = keep_by_elements(
        input_file, elements, mode
    )

    mode_label = "TODOS" if mode == "all" else "ALGUNO DE"
    print(f"Elementos buscados: {', '.join(elements)} (modo: {mode_label})")
    print(f"Estructuras totales en el archivo original: {total_structures}")
    print(f"Estructuras conservadas: {kept_count}")
    print(f"Estructuras eliminadas: {removed_count}")

    if filtered_atoms:
        write(output_file, filtered_atoms)
        print(f"Se guardaron {kept_count} configuraciones en: {output_file}")
    else:
        print("Ninguna configuración cumplió el criterio, no se generó archivo de salida.")

if __name__ == "__main__":
    main()