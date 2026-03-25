#!/usr/bin/env python3
"""
Script para añadir átomos shell a estructuras SiO2.
Verifica que N(O) = 2*N(Si) y añade shells para los oxígenos.
"""

import sys
import argparse
import subprocess
from ase.io import read, write
from ase import Atoms, Atom
from ase.data import atomic_numbers
import numpy as np


def verify_sio2_stoichiometry(atoms):
    """
    Verifica que la estructura tenga el ratio correcto Si:O = 1:2
    
    Args:
        atoms: objeto ASE Atoms
        
    Returns:
        tuple: (n_si, n_o, is_valid)
    """
    symbols = atoms.get_chemical_symbols()
    n_si = symbols.count('Si')
    n_o = symbols.count('O')
    
    is_valid = (n_o == 2 * n_si)
    
    return n_si, n_o, is_valid


def add_shells_to_oxygen(atoms, shell_symbol='Sh', shell_offset=0.0):
    """
    Añade átomos shell en las posiciones de los oxígenos.
    
    Args:
        atoms: objeto ASE Atoms con la estructura original
        shell_symbol: símbolo para los átomos shell (default: 'Sh')
        shell_offset: desplazamiento en Angstroms desde la posición del oxígeno (default: 0.0)
        
    Returns:
        nuevo objeto Atoms con los shells añadidos
    """
    # Crear nueva estructura desde la original
    new_atoms = atoms.copy()
    
    # Encontrar índices de oxígenos y añadir shells
    for i, symbol in enumerate(atoms.get_chemical_symbols()):
        if symbol == 'O':
            # Posición del shell (puede tener un offset)
            shell_pos = atoms.positions[i].copy() + np.array([shell_offset, 0.0, 0.0])
            
            # Intentar crear átomo con el símbolo dado
            # Si no es reconocido por ASE, usar número atómico 0 (átomo dummy)
            try:
                if shell_symbol in atomic_numbers:
                    shell_atom = Atom(shell_symbol, shell_pos)
                else:
                    # Usar número atómico 0 para símbolos personalizados
                    shell_atom = Atom(0, shell_pos)
                    # Cambiar manualmente el símbolo
                    new_atoms.append(shell_atom)
                    new_atoms[-1].symbol = shell_symbol
                    continue
            except:
                shell_atom = Atom(0, shell_pos)
                new_atoms.append(shell_atom)
                new_atoms[-1].symbol = shell_symbol
                continue
            
            new_atoms.append(shell_atom)
    
    return new_atoms


def main():
    parser = argparse.ArgumentParser(
        description='Añade átomos shell a estructuras SiO2 en formato CIF'
    )
    parser.add_argument(
        'input_cif',
        type=str,
        help='Archivo CIF de entrada con la estructura SiO2'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Archivo de salida (default: sobrescribe el archivo de entrada)'
    )
    parser.add_argument(
        '--shell-symbol',
        type=str,
        default='X',
        help='Símbolo para los átomos shell (default: X para dummy atoms, puede usar Sh u otros)'
    )
    parser.add_argument(
        '--shell-offset',
        type=float,
        default=0.0,
        help='Desplazamiento del shell desde el core en Angstroms (default: 0.0)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='cif',
        choices=['cif', 'xyz', 'vasp', 'lammps-data'],
        help='Formato del archivo de salida (default: cif)'
    )
    
    args = parser.parse_args()
    
    # Leer estructura
    print(f"Leyendo estructura desde {args.input_cif}...")
    try:
        atoms = read(args.input_cif)
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        sys.exit(1)
    
    # Verificar estequiometría
    print("\nVerificando estequiometría...")
    n_si, n_o, is_valid = verify_sio2_stoichiometry(atoms)
    
    print(f"  Átomos de Si: {n_si}")
    print(f"  Átomos de O:  {n_o}")
    print(f"  Ratio O/Si:   {n_o/n_si if n_si > 0 else 0:.2f}")
    
    if not is_valid:
        print(f"\n¡ERROR! La estructura NO cumple con N(O) = 2*N(Si)")
        print(f"Se esperaban {2*n_si} átomos de O, pero se encontraron {n_o}")
        sys.exit(1)
    
    print("✓ La estructura cumple con N(O) = 2*N(Si)")
    
    # Añadir shells
    print(f"\nAñadiendo átomos shell (símbolo: {args.shell_symbol})...")
    atoms_with_shells = add_shells_to_oxygen(
        atoms, 
        shell_symbol=args.shell_symbol,
        shell_offset=args.shell_offset
    )
    
    print(f"  Átomos originales: {len(atoms)}")
    print(f"  Átomos con shells: {len(atoms_with_shells)}")
    print(f"  Shells añadidos:   {len(atoms_with_shells) - len(atoms)}")
    
    # Determinar nombre de salida
    if args.output is None:
        args.output = args.input_cif
    
    # Guardar estructura
    print(f"\nGuardando estructura en {args.output}...")
    try:
        write(args.output, atoms_with_shells, format=args.format)
        print("✓ Estructura guardada exitosamente")
        
        # Reemplazar X por shO usando sed (ASE no reconoce shO como elemento)
        if args.shell_symbol == 'X':
            print("\nReemplazando 'X' por 'shO' en el archivo...")
            sed_cmd = f"sed -i 's/X/shO/g' {args.output}"
            result = subprocess.run(sed_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ Reemplazo completado exitosamente")
            else:
                print(f"⚠ Advertencia: Error en el reemplazo con sed: {result.stderr}")
    except Exception as e:
        print(f"Error al guardar el archivo: {e}")
        sys.exit(1)
    
    # Resumen final
    print("\n" + "="*50)
    print("RESUMEN:")
    print("="*50)
    shell_label = 'shO' if args.shell_symbol == 'X' else args.shell_symbol
    print(f"Estructura original:  {n_si} Si + {n_o} O = {len(atoms)} átomos")
    print(f"Estructura con shells: {n_si} Si + {n_o} O + {n_o} {shell_label} = {len(atoms_with_shells)} átomos")
    print(f"Archivo de salida:    {args.output}")
    print("="*50)


if __name__ == "__main__":
    main()
