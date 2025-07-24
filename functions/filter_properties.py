#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
from ase.io import read, write
import numpy as np
import os

def write_filtered_frames(output_file, valid_frames):
    """
    Función auxiliar para escribir frames filtrados a un archivo.
    
    Args:
        output_file (str): Ruta del archivo de salida
        valid_frames (list): Lista de frames válidos
    """
    with open(output_file, 'w') as f_out:
        for i, atoms in enumerate(valid_frames):
            if i == 0:
                write(f_out, atoms, format='extxyz')
            else:
                write(f_out, atoms, format='extxyz', append=True)

def print_statistics(property_name, input_file, output_file, threshold, 
                    total_frames, kept_frames, units, extra_info=None):
    """
    Función auxiliar para imprimir estadísticas de filtrado.
    
    Args:
        property_name (str): Nombre de la propiedad filtrada
        input_file (str): Archivo de entrada
        output_file (str): Archivo de salida
        threshold (float): Umbral usado
        total_frames (int): Total de frames procesados
        kept_frames (int): Frames conservados
        units (str): Unidades de la propiedad
        extra_info (dict): Información adicional específica por propiedad
    """
    print(f"\n{'='*50}")
    print(f"{property_name.upper()} FILTERING STATISTICS")
    print("="*50)
    
    if extra_info:
        # Para energía - mostrar información adicional
        print(f"Total configurations read: {total_frames}")
        print(f"Configurations below threshold ({threshold} {units}): {extra_info['below']}")
        print(f"Configurations above threshold ({threshold} {units}): {extra_info['above']}")
        print(f"Largest group saved: {extra_info['group_type']} threshold")
        print(f"✓ Main file saved: {output_file}")
        print(f"Total configurations saved: {kept_frames}")
        
        # Mostrar información sobre el archivo descartado
        if extra_info.get('discarded_filename'):
            print(f"✓ Discarded file saved: {extra_info['discarded_filename']}")
            print(f"Discarded configurations ({extra_info['discarded_type']} threshold): {extra_info['discarded_count']}")
        else:
            print("✗ No discarded configurations to save")
    else:
        # Para fuerzas y stress
        print(f"Archivo de entrada: {input_file}")
        if output_file:
            print(f"Archivo filtrado: {output_file}")
        else:
            print("No se creó archivo de salida (ningún frame válido)")
        print(f"Umbral de {property_name.lower()}: {threshold} {units}")
        print(f"Frames totales: {total_frames}")
        print(f"Frames conservados: {kept_frames} ({kept_frames/total_frames:.1%})" if total_frames > 0 else "Frames conservados: 0")
        print(f"Frames descartados: {total_frames - kept_frames}")
    
    print("="*50)

def filter_by_property(input_file, threshold, property_type):
    """
    Función genérica para filtrar por fuerzas o stress.
    
    Args:
        input_file (str): Archivo de entrada
        threshold (float): Umbral máximo
        property_type (str): 'forces' o 'stress'
    
    Returns:
        str: Archivo de salida generado
    """
    # Configuración específica por propiedad
    config = {
        'forces': {
            'suffix': '-forces-filtered.xyz',
            'getter': lambda atoms: atoms.get_forces(),
            'units': 'eV/Å',
            'name': 'fuerza',
            'property_name': 'forces'
        },
        'stress': {
            'suffix': '-stress-filtered.xyz',
            'getter': lambda atoms: atoms.get_stress(),
            'units': 'eV/Å³',
            'name': 'stress',
            'property_name': 'stress'
        }
    }
    
    prop_config = config[property_type]
    
    # Generar nombre de archivo de salida
    base_name, ext = os.path.splitext(input_file)
    output_file = f"{base_name}{prop_config['suffix']}"
    
    # Contadores para estadísticas
    total_frames = 0
    kept_frames = 0
    valid_frames = []
    
    # Leer el archivo frame por frame
    for atoms in read(input_file, index=':', format='extxyz'):
        total_frames += 1
        
        try:
            property_data = prop_config['getter'](atoms)
        except AttributeError:
            print(f"Advertencia: Frame {total_frames} no contiene {prop_config['name']} válidas. Descartado.")
            continue

        # Calcular norma total
        total_norm = np.linalg.norm(property_data)
        
        # Agregar a la lista si pasa el filtro
        if total_norm <= threshold:
            valid_frames.append(atoms)
            kept_frames += 1
        else:
            print(f"Descartado frame {total_frames}: Norma total de {prop_config['name']} = {total_norm:.2f} {prop_config['units']}")
    
    # Crear archivo de salida si hay frames válidos
    if kept_frames > 0:
        write_filtered_frames(output_file, valid_frames)
    else:
        print(f"No se encontraron frames válidos. No se creará archivo de salida.")
        output_file = None
    
    # Mostrar estadísticas
    print_statistics(prop_config['property_name'], input_file, output_file, threshold, 
                    total_frames, kept_frames, prop_config['units'])
    
    return output_file

def filter_by_energy(xyz_file, energy_threshold):
    """
    Filtra configuraciones por energía por átomo y guarda el grupo más grande.
    
    Args:
        xyz_file (str): Ruta al archivo .xyz de entrada
        energy_threshold (float): Umbral de energía por átomo (eV/átomo)
    
    Returns:
        str: Ruta al archivo de salida generado
    """
    # Leer el archivo .xyz
    data = read(xyz_file, index=':')
    base_name, ext = os.path.splitext(xyz_file)
    
    # Separar configuraciones por energía
    below_threshold = []
    above_threshold = []
    
    for atoms in data:
        energy_per_atom = atoms.get_potential_energy() / len(atoms)
        if energy_per_atom < energy_threshold:
            below_threshold.append(atoms)
        else:
            above_threshold.append(atoms)
    
    # Determinar el conjunto más grande
    if len(below_threshold) > len(above_threshold):
        largest_set = below_threshold
        group_type = "Below"
    else:
        largest_set = above_threshold
        group_type = "Above"
    
    # Guardar el conjunto más grande en un archivo .xyz
    output_filename = f'{base_name}-energy-filtered.xyz'
    write(output_filename, largest_set)
    
def filter_by_energy(xyz_file, energy_threshold):
    """
    Filtra configuraciones por energía por átomo y guarda el grupo más grande.
    También guarda el grupo descartado en un archivo separado.
    
    Args:
        xyz_file (str): Ruta al archivo .xyz de entrada
        energy_threshold (float): Umbral de energía por átomo (eV/átomo)
    
    Returns:
        str: Ruta al archivo de salida principal (grupo más grande)
    """
    # Leer el archivo .xyz
    data = read(xyz_file, index=':')
    base_name, ext = os.path.splitext(xyz_file)
    
    # Separar configuraciones por energía
    below_threshold = []
    above_threshold = []
    
    for atoms in data:
        energy_per_atom = atoms.get_potential_energy() / len(atoms)
        if energy_per_atom < energy_threshold:
            below_threshold.append(atoms)
        else:
            above_threshold.append(atoms)
    
    # Determinar el conjunto más grande y el descartado
    if len(below_threshold) > len(above_threshold):
        largest_set = below_threshold
        discarded_set = above_threshold
        group_type = "Below"
        discarded_type = "Above"
    else:
        largest_set = above_threshold
        discarded_set = below_threshold
        group_type = "Above"
        discarded_type = "Below"
    
    # Guardar el conjunto más grande en un archivo .xyz
    output_filename = f'{base_name}-energy-filtered.xyz'
    write(output_filename, largest_set)
    
    # Guardar el conjunto descartado si no está vacío
    discarded_filename = None
    if len(discarded_set) > 0:
        discarded_filename = f'{base_name}-energy-discarded.xyz'
        write(discarded_filename, discarded_set)
    
    # Mostrar estadísticas usando la función auxiliar
    extra_info = {
        'below': len(below_threshold),
        'above': len(above_threshold),
        'group_type': group_type,
        'discarded_filename': discarded_filename,
        'discarded_type': discarded_type,
        'discarded_count': len(discarded_set)
    }
    
    print_statistics("energy", xyz_file, output_filename, energy_threshold, 
                    len(data), len(largest_set), "eV/atom", extra_info)
    
    return output_filename

def filter_by_forces(input_file, max_force_threshold):
    """Filtra frames por norma total de fuerzas."""
    return filter_by_property(input_file, max_force_threshold, 'forces')

def filter_by_stress(input_file, max_stress_threshold):
    """Filtra frames por norma total de stress."""
    return filter_by_property(input_file, max_stress_threshold, 'stress')

def show_help():
    """Muestra el mensaje de ayuda."""
    print("="*80)
    print("                    UNIFIED PROPERTIES FILTER FOR XYZ FILES")
    print("="*80)
    print()
    print("DESCRIPTION:")
    print("  This script filters atomic configurations from XYZ files based on different")
    print("  properties: energy, forces, or stress. Choose the property type and")
    print("  specify the threshold value.")
    print()
    print("USAGE:")
    print("  python filter_properties.py <property_type> <threshold> <file.xyz>")
    print()
    print("PROPERTY TYPES:")
    print("  energy      Filter by energy per atom (eV/atom)")
    print("  forces      Filter by total force norm (eV/Å)")
    print("  stress    Filter by total stress norm (eV/Å³)")
    print()
    print("ARGUMENTS:")
    print("  property_type       Type of property to filter (energy/forces/stress)")
    print("  threshold           Threshold value for the selected property")
    print("  file.xyz            Input XYZ file with atomic configurations")
    print()
    print("EXAMPLES:")
    print("  python filter_properties.py energy -2.5 configurations.xyz")
    print("  python filter_properties.py forces 100 trajectory.extxyz")
    print("  python filter_properties.py stress 50 md_data.extxyz")
    print()
    print("OUTPUT:")
    print("  - Energy: <original_name>-energy-filtered.xyz (largest group)")
    print("            <original_name>-energy-discarded.xyz (discarded group)")
    print("  - Forces: <original_name>-forces-filtered.xyz (below threshold)")
    print("  - stress: <original_name>-stress-filtered.xyz (below threshold)")
    print("  - Statistical information displayed on screen")
    print()
    print("NOTES:")
    print("  • Energy filtering saves the largest group (above or below threshold)")
    print("    and also saves the discarded group in a separate file")
    print("  • Forces and stress filtering save frames below the threshold")
    print("  • Files must contain the corresponding property information")
    print("  • Compatible with XYZ files generated by ASE, VASP, CP2K, etc.")
    print()
    print("="*80)

def parse_arguments():
    """Parsea y valida los argumentos de línea de comandos."""
    property_type = sys.argv[1].lower()
    
    try:
        threshold = float(sys.argv[2])
    except ValueError:
        print("Error: El umbral debe ser un número")
        sys.exit(1)
    
    input_file = sys.argv[3]
    
    if not os.path.isfile(input_file):
        print(f"Error: El archivo {input_file} no existe")
        sys.exit(1)
    
    return property_type, threshold, input_file

def main():
    # Check arguments and show help
    if len(sys.argv) < 4 or sys.argv[1] in ['--help', '-h', '--helop']:
        show_help()
        sys.exit(1)
    
    # Parse and validate arguments
    property_type, threshold, input_file = parse_arguments()
    
    # Property configuration
    filters = {
        "energy": lambda f, t: filter_by_energy(f, t),
        "forces": lambda f, t: filter_by_forces(f, t),
        "stress": lambda f, t: filter_by_stress(f, t)
    }
    
    units = {"energy": "eV/atom", "forces": "eV/Å", "stress": "eV/Å³"}
    
    # Execute filter
    if property_type not in filters:
        print(f"Error: Tipo de propiedad '{property_type}' no reconocido.")
        print("Tipos válidos: energy, forces, stress")
        sys.exit(1)
    
    print(f"\nFiltrando por {property_type} con umbral: {threshold} {units[property_type]}")
    output_file = filters[property_type](input_file, threshold)
    
    # Final message
    if output_file:
        print(f"\n🎉 Filtrado completado exitosamente!")
        print(f"Archivo de salida: {output_file}")
    else:
        print(f"\n❌ No se pudo crear archivo de salida.")

if __name__ == "__main__":
    main()
