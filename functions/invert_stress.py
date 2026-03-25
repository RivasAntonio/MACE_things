#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Invert the sign of stress-related properties in trajectory frames."""

from ase.io import read, write
import numpy as np
import sys

def invert_stress(input_file, output_file):
    print(f"Leyendo {input_file}...")
    
    # index=':' lee todas las configuraciones (trajectory) no solo la primera
    atoms_list = read(input_file, index=':')
    
    count = 0
    skipped = 0
    keys_modified = {}  # Diccionario para contar cuántas veces se modifica cada clave
    
    for i, at in enumerate(atoms_list):
        # Debug: mostrar qué tiene la primera estructura
        if i == 0:
            print(f"Claves en at.info: {list(at.info.keys())}")
            print(f"Claves en at.arrays: {list(at.arrays.keys())}")
        
        # Buscar TODAS las claves que contengan 'stress' (case insensitive)
        # EXCEPTO 'REF_stress' que no se modifica
        stress_found = False
        
        # Buscar en at.info
        for key in list(at.info.keys()):
            if 'stress' in key.lower() and key.lower() != 'ref_stress':
                at.info[key] = -1.0 * np.array(at.info[key])
                stress_found = True
                keys_modified[f"info['{key}']"] = keys_modified.get(f"info['{key}']", 0) + 1
        
        # Buscar en at.arrays
        for key in list(at.arrays.keys()):
            if 'stress' in key.lower() and key.lower() != 'ref_stress':
                at.arrays[key] = -1.0 * at.arrays[key]
                stress_found = True
                keys_modified[f"arrays['{key}']"] = keys_modified.get(f"arrays['{key}']", 0) + 1
        
        if stress_found:
            count += 1
        else:
            skipped += 1

    print(f"\nResumen:")
    print(f"  - Se invirtió el signo del stress en {count} configuraciones.")
    print(f"  - Se omitieron {skipped} configuraciones sin stress.")
    
    if keys_modified:
        print(f"\nClaves modificadas:")
        for key, n in keys_modified.items():
            print(f"  - {key}: {n} veces")
    
    # Guardamos el nuevo archivo
    # write_results=False evita duplicar información si ya está en info
    write(output_file, atoms_list)
    print(f"Guardado en {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python invert_stress.py input.xyz output.xyz")
    else:
        invert_stress(sys.argv[1], sys.argv[2])
