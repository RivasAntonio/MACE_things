#!/usr/bin/env python3
"""
Script para separar configuraciones según si su info['origin'] 
está en una lista de origins proporcionada en un archivo de texto.

Uso: python separate_by_origin_list.py archivo.xyz lista_origins.txt
"""

import sys
import os
from ase.io import read, write


def load_origin_list(txt_file):
    """
    Carga la lista de origins desde un archivo de texto.
    Un origin por línea.
    """
    origins = set()
    with open(txt_file, 'r') as f:
        for line in f:
            origin = line.strip()
            if origin:  # Ignorar líneas vacías
                origins.add(origin)
    return origins


def separate_by_origin_list(input_file, txt_file):
    """
    Separa las configuraciones en dos archivos:
    - uno con las que tienen origin en la lista
    - otro con las que no están en la lista
    """
    # Cargar lista de origins
    print(f"Cargando lista de origins de {txt_file}...")
    origin_list = load_origin_list(txt_file)
    print(f"Origins en la lista: {len(origin_list)}")
    
    # Leer configuraciones
    print(f"Leyendo configuraciones de {input_file}...")
    configurations = read(input_file, index=':')
    print(f"Total de configuraciones: {len(configurations)}")
    
    # Separar configuraciones
    in_list_configs = []
    not_in_list_configs = []
    
    for atoms in configurations:
        origin = atoms.info.get('origin', '')
        if origin in origin_list:
            in_list_configs.append(atoms)
        else:
            not_in_list_configs.append(atoms)
    
    print(f"Configuraciones en la lista: {len(in_list_configs)}")
    print(f"Configuraciones fuera de la lista: {len(not_in_list_configs)}")
    
    # Generar nombres de archivos de salida
    base_name = os.path.splitext(input_file)[0]
    list_name = os.path.splitext(os.path.basename(txt_file))[0]
    in_list_output = f"{base_name}_in_{list_name}.xyz"
    not_in_list_output = f"{base_name}_not_in_{list_name}.xyz"
    
    # Escribir archivos de salida
    if in_list_configs:
        write(in_list_output, in_list_configs)
        print(f"Configuraciones en lista guardadas en: {in_list_output}")
    else:
        print("No se encontraron configuraciones en la lista")
    
    if not_in_list_configs:
        write(not_in_list_output, not_in_list_configs)
        print(f"Configuraciones fuera de lista guardadas en: {not_in_list_output}")
    else:
        print("No se encontraron configuraciones fuera de la lista")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python separate_by_origin_list.py archivo.xyz lista_origins.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    txt_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: El archivo {input_file} no existe")
        sys.exit(1)
    
    if not os.path.exists(txt_file):
        print(f"Error: El archivo {txt_file} no existe")
        sys.exit(1)
    
    separate_by_origin_list(input_file, txt_file)
