#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para ordenar estructuras de un archivo .extxyz por energía total

Este script lee un archivo .extxyz usando ASE, ordena las estructuras de menor a mayor energía total
y guarda las estructuras ordenadas en un nuevo archivo.
"""

import os
import sys
import argparse
from ase.io import read, write
import numpy as np

def sort_structures_by_energy(input_file, output_file=None, ascending=True, verbose=False):
    """
    Lee un archivo .extxyz, ordena las estructuras por energía total y guarda un nuevo archivo.
    
    Argumentos:
        input_file (str): Ruta al archivo .extxyz de entrada
        output_file (str, opcional): Ruta para el archivo de salida. Si es None,
                                     se genera automáticamente a partir del nombre original
        ascending (bool): True para ordenar de menor a mayor energía total, False para mayor a menor
        verbose (bool): Si es True, muestra información detallada durante el proceso
    
    Returns:
        str: Ruta del archivo de salida generado
    """
    # Determinar nombre del archivo de salida si no se proporciona
    if output_file is None:
        base_name, ext = os.path.splitext(input_file)
        output_file = f"{base_name}_sorted{ext}"
    
    if verbose:
        print(f"Leyendo estructuras desde: {input_file}")
    
    try:
        # Leer todas las estructuras del archivo especificando explícitamente el formato
        structures = read(input_file, index=":", format="extxyz")
        
        if not structures:
            print(f"ERROR: No se encontraron estructuras en {input_file}")
            return None
        
        if verbose:
            print(f"Leídas {len(structures)} estructuras")
        
        # Función para obtener la energía de una estructura de manera confiable
        def get_energy(atoms):
            # Intentar obtener la energía de diferentes maneras
            # 1. De la propiedad directa get_potential_energy
            energy = atoms.get_potential_energy()
            
            # 2. Del diccionario info si existe
            if energy == 0 and 'energy' in atoms.info:
                energy = atoms.info['energy']
            
            # 3. Del diccionario info.calc si existe
            if energy == 0 and hasattr(atoms, 'calc') and atoms.calc is not None:
                try:
                    energy = atoms.calc.get_potential_energy()
                except:
                    pass
                    
            return energy
        
        # Verificar que las estructuras tienen energía
        missing_energy = False
        for i, atoms in enumerate(structures):
            energy = get_energy(atoms)
            # Solo consideramos que falta la energía si es exactamente 0 y no hay ninguna propiedad energy
            if energy == 0 and 'energy' not in atoms.info and not (hasattr(atoms, 'calc') and atoms.calc is not None):
                missing_energy = True
                if verbose:
                    print(f"ADVERTENCIA: La estructura {i+1} no tiene información de energía")
        
        if missing_energy and verbose:
            print("ADVERTENCIA: Algunas estructuras podrían no tener energía definida correctamente")
        
        # Obtener las energías totales (sin normalizar por átomo)
        energies = np.array([get_energy(atoms) for atoms in structures])
        
        # Ordenar índices por energía total
        sorted_indices = np.argsort(energies)
        if not ascending:
            sorted_indices = sorted_indices[::-1]
        
        # Ordenar las estructuras
        sorted_structures = [structures[i] for i in sorted_indices]
        
        # Guardar archivo ordenado - especificar explícitamente el formato
        write(output_file, sorted_structures, format="extxyz")
        
        if verbose:
            print(f"Archivo ordenado guardado como: {output_file}")
            print("\nResumen de energías (primeras 5 y últimas 5 estructuras):")
            print("-" * 60)
            print("Posición | Energía total (eV) | Núm. átomos | Energía (eV/átomo)")
            print("-" * 60)
            
            # Mostrar las primeras 5 estructuras
            for i in range(min(5, len(sorted_structures))):
                atoms = sorted_structures[i]
                e_total = get_energy(atoms)
                e_per_atom = e_total / len(atoms)
                print(f"{i+1:8d} | {e_total:17.6f} | {len(atoms):11d} | {e_per_atom:17.6f}")
            
            # Si hay más de 10 estructuras, mostrar separador
            if len(sorted_structures) > 10:
                print("...")
                
            # Mostrar las últimas 5 estructuras
            for i in range(max(5, len(sorted_structures)-5), len(sorted_structures)):
                atoms = sorted_structures[i]
                e_total = get_energy(atoms)
                e_per_atom = e_total / len(atoms)
                print(f"{i+1:8d} | {e_total:17.6f} | {len(atoms):11d} | {e_per_atom:17.6f}")
        
        return output_file
    
    except Exception as e:
        print(f"ERROR: No se pudieron ordenar las estructuras: {e}")
        return None

def main():
    # Configurar el parser de argumentos
    parser = argparse.ArgumentParser(
        description='Ordena estructuras de un archivo .extxyz por energía total',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python sort_structures_by_energy.py estructuras.extxyz
  python sort_structures_by_energy.py estructuras.extxyz -o ordenadas.extxyz
  python sort_structures_by_energy.py estructuras.extxyz -d -v
        """
    )
    
    parser.add_argument('input_file', help='Archivo .extxyz de entrada')
    parser.add_argument('-o', '--output', help='Archivo de salida (opcional)')
    parser.add_argument('-d', '--descending', action='store_true', 
                        help='Ordenar de mayor a menor energía')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Mostrar información detallada durante el proceso')
    parser.add_argument('--debug', action='store_true',
                        help='Mostrar información de diagnóstico adicional')
    
    args = parser.parse_args()
    
    # Mostrar información de diagnóstico si se solicita
    if args.debug:
        print("Modo de diagnóstico activado")
        try:
            # Leer primera estructura para examinar
            first_atoms = read(args.input_file, index=0, format="extxyz")
            
            print("\nInformación de la primera estructura:")
            print(f"Número de átomos: {len(first_atoms)}")
            print(f"Propiedades disponibles en atoms.info: {list(first_atoms.info.keys())}")
            print(f"Posiciones: {first_atoms.positions.shape}")
            
            # Intentar diferentes métodos para obtener la energía
            print("\nIntentando obtener energía mediante diferentes métodos:")
            print(f"1. atoms.get_potential_energy(): {first_atoms.get_potential_energy()}")
            print(f"2. atoms.info.get('energy'): {first_atoms.info.get('energy', 'No disponible')}")
            if hasattr(first_atoms, 'calc') and first_atoms.calc is not None:
                print(f"3. atoms.calc.get_potential_energy(): {first_atoms.calc.get_potential_energy()}")
            else:
                print("3. atoms.calc: No disponible")
                
            # Examinar formato del archivo
            print("\nExaminando formato del archivo:")
            with open(args.input_file, 'r') as f:
                first_lines = [next(f) for _ in range(min(10, len(first_atoms) + 2))]
            print("Primeras líneas del archivo:")
            for i, line in enumerate(first_lines):
                print(f"{i+1}: {line.strip()}")
        
        except Exception as e:
            print(f"Error durante el diagnóstico: {e}")
    
    # Llamar a la función principal
    sort_structures_by_energy(
        args.input_file, 
        args.output,
        ascending=not args.descending,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
