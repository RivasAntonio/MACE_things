#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from ase.io import read
from ase.data import chemical_symbols, atomic_numbers
import numpy as np
import sys
import os

def analizar_xyz(ruta_archivo):
    try:
        # Leer todas las estructuras del archivo .xyz
        estructuras = read(ruta_archivo, index=":")
    except Exception as e:
        print(f"❌ Error al leer el archivo: {e}")
        sys.exit(1)

    if not estructuras:
        print("⚠️ No se encontraron estructuras en el archivo.")
        sys.exit(1)

    # Contar los átomos en cada estructura
    num_atomos = [len(atoms) for atoms in estructuras]

    # Calcular estadísticas
    media = np.mean(num_atomos)
    minimo = np.min(num_atomos)
    maximo = np.max(num_atomos)

    # Contar átomos por tipo en todas las estructuras
    atom_counts = {}  # Para contar el número total de átomos de cada elemento
    struct_counts = {}  # Para contar el número de estructuras que contienen cada elemento
    
    for atoms in estructuras:
        # Obtener los símbolos químicos únicos en esta estructura
        symbols_in_struct = set(atoms.get_chemical_symbols())
        
        # Actualizar el conteo de estructuras para cada elemento único
        for symbol in symbols_in_struct:
            struct_counts[symbol] = struct_counts.get(symbol, 0) + 1
        
        # Actualizar el conteo total de átomos
        for symbol in atoms.get_chemical_symbols():
            atom_counts[symbol] = atom_counts.get(symbol, 0) + 1

    total_atoms = sum(atom_counts.values())
    total_estructuras = len(estructuras)

    # Mostrar resultados
    print(f" Archivo analizado: {ruta_archivo}")
    print(f" Número total de estructuras: {total_estructuras}")
    print(f" Número medio de átomos por estructura: {media:.2f}")
    print(f" Número mínimo de átomos en una estructura: {minimo}")
    print(f" Número máximo de átomos en una estructura: {maximo}")
    print("\n Conteo de átomos y presencia en estructuras:")
    print("="*50)
    # Ordenar los símbolos por número atómico
    sorted_symbols = sorted(atom_counts.keys(), key=lambda s: atomic_numbers[s])
    for symbol in sorted_symbols:
        count = atom_counts[symbol]
        struct_count = struct_counts[symbol]
        atomic_number = atomic_numbers[symbol]
        porcentaje_atomos = (count / total_atoms) * 100 if total_atoms > 0 else 0
        porcentaje_estructuras = (struct_count / total_estructuras) * 100
        print(f" {symbol} (Z={atomic_number}): {count} átomos ({porcentaje_atomos:.2f}% del total), presente en {struct_count}/{total_estructuras} estructuras ({porcentaje_estructuras:.2f}%)")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analiza un archivo .xyz y muestra estadísticas de número de átomos.")
    parser.add_argument("archivo", help="Ruta al archivo .xyz")

    args = parser.parse_args()

    if not os.path.isfile(args.archivo):
        print(f"❌ El archivo '{args.archivo}' no existe.")
        sys.exit(1)

    analizar_xyz(args.archivo)

