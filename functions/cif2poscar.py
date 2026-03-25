#!/usr/bin/env python3
"""Convert CIF files into POSCAR files with optional POTCAR-based element ordering."""

import os
import sys
import numpy as np
from ase.io import read
from collections import Counter

def get_potcar_order(potcar_path="POTCAR"):
    elements = []
    with open(potcar_path, "r") as f:
        for line in f:
            if line.startswith("   VRHFIN"):
                element = line.split("=")[1].split(":")[0].strip()
                elements.append(element)
    return elements

def reorder_atoms(atoms, order):
    new_indices = []
    symbols = atoms.get_chemical_symbols()
    for element in order:
        new_indices += [i for i, sym in enumerate(symbols) if sym == element]
    return atoms[new_indices]

def count_elements(atoms, element_order):
    symbols = atoms.get_chemical_symbols()
    counts = Counter(symbols)
    return [counts[el] for el in element_order]

def write_poscar_manual(atoms, element_order, filename, selective=False):
    cell = atoms.get_cell()
    coords = atoms.get_scaled_positions()

    with open(filename, "w") as f:
        f.write("POSCAR generado manualmente\n")
        f.write(" 1.00000000000000\n")
        for vec in cell:
            f.write(" {:>12.10f} {:>12.10f} {:>12.10f}\n".format(*vec))

        f.write(" " + "  ".join(element_order) + "\n")
        counts = count_elements(atoms, element_order)
        f.write(" " + "  ".join(str(c) for c in counts) + "\n")

        if selective:
            f.write("Selective dynamics\n")
        f.write("Direct\n")

        symbols = atoms.get_chemical_symbols()
        scaled = atoms.get_scaled_positions()

        for i, atom in enumerate(atoms):
            pos = scaled[i]
            if selective:
                flags = "F F F" if i == 0 else "T T T"
                f.write(" {:>10.5f} {:>10.5f} {:>10.5f}   {}\n".format(pos[0], pos[1], pos[2], flags))
            else:
                f.write(" {:>10.5f} {:>10.5f} {:>10.5f}\n".format(pos[0], pos[1], pos[2]))

def cif_to_poscar(cif_path, potcar_path="POTCAR", selective=False):
    if not os.path.isfile(potcar_path):
        print("Error: no se encontró el archivo POTCAR en el directorio actual.")
        sys.exit(1)

    atoms = read(cif_path)
    potcar_order = get_potcar_order(potcar_path)
    atoms = reorder_atoms(atoms, potcar_order)

    base = os.path.splitext(os.path.basename(cif_path))[0]
    output_path = f"POSCAR_{base}"
    write_poscar_manual(atoms, potcar_order, output_path, selective=selective)

    print(f"POSCAR escrito en: {output_path} {'(con Selective Dynamics)' if selective else ''}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python cif2poscar.py archivo.cif [activar_SD]")
        sys.exit(1)

    cif_files = sys.argv[1:]
    #selective_dynamics = len(sys.argv) >   # cualquier argumento adicional activa SD
    for cif_file in cif_files:
        cif_to_poscar(cif_file)

