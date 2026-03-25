#!/usr/bin/env python3
"""
Script para crear superceldas a partir de un archivo de estructura.
Usa ASE para leer/escribir diferentes formatos (POSCAR, xyz, cif, etc.)
"""

import argparse
import sys
import os
from ase.io import read, write
from collections import Counter

def reorder_poscar_format(atoms, file_path):
    """
    Reordena y escribe un archivo POSCAR en el formato correcto,
    agrupando átomos por elemento.
    
    Args:
        atoms: Objeto ASE Atoms
        file_path: Ruta donde guardar el POSCAR
    """
    # Contar número de átomos de cada elemento
    symbols = atoms.get_chemical_symbols()
    count = Counter(symbols)
    
    # Mantener orden de aparición de los elementos
    elements_sorted = []
    for s in symbols:
        if s not in elements_sorted:
            elements_sorted.append(s)
    
    # Primera línea: Elemento+Número
    first_line = " ".join(f"{el}{count[el]}" for el in elements_sorted)
    
    # Línea de elementos y cantidades
    elements_line = " ".join(elements_sorted)
    numbers_line = " ".join(str(count[el]) for el in elements_sorted)
    
    # Convertir posiciones a fraccionarias
    frac_positions = atoms.get_scaled_positions()
    
    # Reordenar posiciones según elements_sorted
    positions_dict = {el: [] for el in elements_sorted}
    for i, s in enumerate(symbols):
        positions_dict[s].append(frac_positions[i])
    
    # Escribir el POSCAR
    with open(file_path, "w") as f:
        f.write(first_line + "\n")
        f.write("1.0\n")
        for vec in atoms.get_cell():
            f.write("   ".join(f"{x:.16f}" for x in vec) + "\n")
        f.write(elements_line + "\n")
        f.write(numbers_line + "\n")
        f.write("Direct\n")
        for el in elements_sorted:
            for pos in positions_dict[el]:
                f.write("   ".join(f"{x:.16f}" for x in pos) + f" {el}\n")

def main():
    parser = argparse.ArgumentParser(
        description='Crea una supercelda multiplicando la celda unitaria en direcciones especificadas.'
    )
    parser.add_argument(
        'archivo_entrada',
        type=str,
        help='Archivo de entrada con la estructura (POSCAR, xyz, cif, etc.)'
    )
    parser.add_argument(
        'archivo_salida',
        type=str,
        help='Archivo de salida para la supercelda'
    )
    parser.add_argument(
        '-x', '--nx',
        type=int,
        default=1,
        help='Multiplicador en dirección x (a). Por defecto: 1'
    )
    parser.add_argument(
        '-y', '--ny',
        type=int,
        default=1,
        help='Multiplicador en dirección y (b). Por defecto: 1'
    )
    parser.add_argument(
        '-z', '--nz',
        type=int,
        default=1,
        help='Multiplicador en dirección z (c). Por defecto: 1'
    )
    parser.add_argument(
        '--formato',
        type=str,
        default=None,
        help='Formato del archivo de salida (vasp, xyz, cif, etc.). Si no se especifica, se infiere de la extensión.'
    )

    args = parser.parse_args()

    # Leer estructura
    try:
        print(f"Leyendo estructura desde: {args.archivo_entrada}")
        atoms = read(args.archivo_entrada)
        print(f"  Átomos originales: {len(atoms)}")
        print(f"  Celda original: {atoms.cell.cellpar()}")
    except Exception as e:
        print(f"Error leyendo el archivo: {e}")
        sys.exit(1)

    # Crear supercelda
    print(f"\nCreando supercelda: {args.nx}x{args.ny}x{args.nz}")
    supercelda = atoms * (args.nx, args.ny, args.nz)
    print(f"  Átomos en supercelda: {len(supercelda)}")
    print(f"  Celda final: {supercelda.cell.cellpar()}")

    # Escribir archivo de salida
    try:
        # Detectar si es formato POSCAR (VASP)
        es_poscar = False
        if args.formato:
            es_poscar = args.formato.lower() in ['vasp', 'poscar']
        else:
            # Inferir del nombre de archivo
            nombre_base = os.path.basename(args.archivo_salida).upper()
            es_poscar = nombre_base in ['POSCAR', 'CONTCAR'] or args.archivo_salida.endswith('.vasp')
        
        if es_poscar:
            print(f"\nGuardando como POSCAR con formato ordenado...")
            reorder_poscar_format(supercelda, args.archivo_salida)
        else:
            if args.formato:
                write(args.archivo_salida, supercelda, format=args.formato)
            else:
                write(args.archivo_salida, supercelda)
        
        print(f"\nSupercelda guardada en: {args.archivo_salida}")
    except Exception as e:
        print(f"Error escribiendo el archivo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
