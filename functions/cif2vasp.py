#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert a CIF structure to POSCAR format using pymatgen."""

import sys
from pymatgen.io.cif import CifParser
from pymatgen.io.vasp import Poscar

def cif2poscar(cif_file):
    try:
        # Read .cif file
        parser = CifParser(cif_file)
        structure = parser.parse_structures(primitive=True)[0]

        # Create poscar
        poscar_name = f"POSCAR_{cif_file.split('.cif')[0]}"
        poscar = Poscar(structure)
        poscar.write_file(poscar_name)
        print(f"{poscar_name} file generated succesfully")

    except Exception as e:
        print(f"Error converting the file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Converting file(s)")
    else:
        for cif_file in sys.argv[1:]:
            cif2poscar(cif_file)


