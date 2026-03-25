#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert CIF crystal structures into LAMMPS data files."""

from pymatgen.core.structure import Structure
from ase.data import atomic_masses, atomic_numbers
import numpy as np
import sys

def cif_to_lammps(cif_file, output_file):
    """
    Convierte un archivo .cif a formato .data de LAMMPS.
    atom_style molecular: atom-id mol-id atom-type x y z

    :param cif_file: Nombre del archivo .cif de entrada.
    :param output_file: Nombre del archivo de salida en formato .data.
    """
    # Cargar la estructura desde el archivo .cif
    structure = Structure.from_file(cif_file)

    # Extraer posiciones y tipos de átomos
    symbols = []
    for site in structure.sites:
        try:
            sym = site.specie.symbol
        except AttributeError:
            sym = max(site.species, key=lambda el: site.species[el]).symbol
        symbols.append(sym)

    positions = structure.cart_coords

    # Orden de especies único, preservando orden de aparición
    specorder = list(dict.fromkeys(symbols))
    # Mapa símbolo -> tipo (1-based)
    type_map = {sym: i + 1 for i, sym in enumerate(specorder)}

    cell = structure.lattice.matrix  # 3x3, filas = vectores a, b, c

    # Ángulos de la celda

    a_vec, b_vec, c_vec = cell[0], cell[1], cell[2]
    a = np.linalg.norm(a_vec)
    b = np.linalg.norm(b_vec)
    c = np.linalg.norm(c_vec)

    # Transformación a triclinic LAMMPS (lower triangular)
    lx = a
    xy = np.dot(b_vec, a_vec / a)
    xz = np.dot(c_vec, a_vec / a)
    ly = np.sqrt(b**2 - xy**2)
    yz = (np.dot(b_vec, c_vec) - xy * xz) / ly
    lz = np.sqrt(c**2 - xz**2 - yz**2)

    with open(output_file, 'w') as f:
        f.write(f"LAMMPS data file from {cif_file}\n\n")
        f.write(f"{len(symbols)} atoms\n")
        f.write(f"{len(specorder)} atom types\n\n")

        f.write(f"0.0 {lx:.10f} xlo xhi\n")
        f.write(f"0.0 {ly:.10f} ylo yhi\n")
        f.write(f"0.0 {lz:.10f} zlo zhi\n")
        if abs(xy) > 1e-10 or abs(xz) > 1e-10 or abs(yz) > 1e-10:
            f.write(f"{xy:.10f} {xz:.10f} {yz:.10f} xy xz yz\n")
        else:
            f.write("0.0 0.0 0.0 xy xz yz\n")
        f.write("\n")

        # Sección Masses con símbolo del elemento como comentario
        f.write("Masses\n\n")
        for sym in specorder:
            atom_num = atomic_numbers[sym]
            mass = atomic_masses[atom_num]
            f.write(f"{type_map[sym]} {mass:.6f}  # {sym}\n")
        f.write("\n")

        # Transformar posiciones al sistema LAMMPS triclinic
        # Matriz de transformación: columnas = vectores a, b, c en cartesianas LAMMPS
        lammps_cell = np.array([
            [lx,  0,  0],
            [xy, ly,  0],
            [xz, yz, lz]
        ])
        # posiciones en coordenadas fraccionarias y luego al sistema LAMMPS
        frac_coords = structure.frac_coords
        lammps_positions = frac_coords @ lammps_cell
        # Sección Atoms: atom-id mol-id atom-type x y z
        f.write("Atoms  \n\n")
        for i, (sym, pos) in enumerate(zip(symbols, lammps_positions)):
            atom_id = i + 1
            mol_id = 1
            atom_type = type_map[sym]
            #f.write(f"{atom_id} {mol_id} {atom_type} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}\n")
            f.write(f"{atom_id}  {atom_type} {pos[0]:.10f} {pos[1]:.10f} {pos[2]:.10f}\n")

    print(f"Archivo {output_file} generado exitosamente.")
    print(f"Tipos de átomos: { {v: k for k, v in type_map.items()} }")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python cif_to_lammps.py archivo.cif [archivo_salida.data]")
    else:
        cif_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "structure.data"
        cif_to_lammps(cif_file, output_file)

