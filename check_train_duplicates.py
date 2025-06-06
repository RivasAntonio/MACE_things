#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para detectar archivos CIF duplicados en uno o más directorios y crear una
estructura de directorios limpia con enlaces simbólicos a los archivos únicos.

Uso:
    python encontrar_cifs_duplicados.py [directorio1] [directorio2] ... [directorioN] [--carpeta-limpia CARPETA]
"""

import sys
def find_duplicate_configurations(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    configurations = []
    current_config = []
    in_config = False
    line_numbers = []  # Para rastrear las líneas de inicio de cada configuración

    for i, line in enumerate(lines):
        if 'Lattice=' in line:
            if current_config:  # Guardar la configuración anterior
                configurations.append((''.join(current_config), line_numbers[-1]))
                current_config = []
            in_config = True
            line_numbers.append(i + 1)  # Guardar la línea de inicio (numeración humana)
        if in_config:
            current_config.append(line)

    # Añadir la última configuración
    if current_config:
        configurations.append((''.join(current_config), line_numbers[-1]))

    # Buscar duplicados
    duplicates = {}
    unique_configurations = []
    seen_configs = set()

    for i, (config, start_line) in enumerate(configurations):
        if config in seen_configs:
            if config in duplicates:
                duplicates[config]['positions'].append(i + 1)  # +1 para numeración humana
                duplicates[config]['lines'].append(start_line)
            else:
                duplicates[config] = {'positions': [i + 1], 'lines': [start_line]}
        else:
            seen_configs.add(config)
            unique_configurations.append(config)

    # Filtrar solo los que tienen duplicados
    duplicates = {k: v for k, v in duplicates.items() if len(v['positions']) > 1}

    if duplicates:
        print("Se encontraron configuraciones duplicadas:")
        for config, data in duplicates.items():
            print(f"Configuración aparece en los bloques: {data['positions']} (líneas: {data['lines']})")
    else:
        print("No se encontraron configuraciones duplicadas.")

    print(len(configurations))
    print(len(duplicates))

    # Sobrescribir el archivo con configuraciones únicas
    with open(file_path, 'w') as file:
        for config in unique_configurations:
            file.write(config)
    
    print(f"Se han guardado {len(unique_configurations)} configuraciones únicas en el archivo {file_path}.")

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Uso: python check_train_duplicates.py <ruta_al_archivo>")
        sys.exit(1)

    file_path = sys.argv[1]
    find_duplicate_configurations(file_path)
# Este script busca configuraciones duplicadas en un archivo de texto y las elimina, dejando solo configuraciones únicas.