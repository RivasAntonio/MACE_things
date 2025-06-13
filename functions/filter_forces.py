import sys
from ase.io import read, write
import numpy as np
import os

def filter_extxyz_by_forces(input_file, max_force_threshold=100):
    """
    Filtra un archivo .extxyz eliminando frames donde alguna fuerza atómica
    supera el umbral especificado (en eV/Å).
    
    Args:
        input_file (str): Ruta al archivo .extxyz de entrada
        max_force_threshold (float): Valor máximo permitido para la fuerza (eV/Å)
    
    Returns:
        str: Ruta al archivo de salida generado
    """
    # Generar nombre de archivo de salida
    base_name, ext = os.path.splitext(input_file)
    output_file = f"{base_name}_filtered.xyz"
    
    # Contadores para estadísticas
    total_frames = 0
    kept_frames = 0
    
    print(f"Procesando {input_file} con umbral {max_force_threshold} eV/Å...")
    
    # Lista para almacenar frames válidos
    valid_frames = []
    
    # Leer el archivo frame por frame
    for atoms in read(input_file, index=':'):
        total_frames += 1
        
        # Buscar 'REF_forces' o 'forces' en los arrays del átomo
        if 'REF_forces' in atoms.arrays:
            forces = atoms.get_array('REF_forces')
        elif 'forces' in atoms.arrays:
            forces = atoms.get_array('forces')
        else:
            continue
        
        # Calcular norma de las fuerzas para cada átomo
        force_magnitudes = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_magnitudes)
        
        # Conservar solo si pasa el filtro
        if max_force <= max_force_threshold:
            valid_frames.append(atoms)
            kept_frames += 1
    
    # Escribir frames válidos al archivo de salida
    if valid_frames:
        for i, atoms in enumerate(valid_frames):
            if i == 0:
                write(output_file, atoms, format='extxyz')
            else:
                write(output_file, atoms, format='extxyz', append=True)
    
    # Reporte final
    print(f"Frames totales: {total_frames}")
    print(f"Frames conservados: {kept_frames}")
    print(f"Frames descartados: {total_frames - kept_frames}")
    print(f"Archivo filtrado: {output_file}")
    
    return output_file

def main():
    # Manejo de argumentos de terminal
    if len(sys.argv) < 3:
        print("Uso: python script.py <umbral_fuerza> <archivo1.extxyz> [archivo2.extxyz ...]")
        print("Ejemplo: python script.py 100 trayectoria1.extxyz trayectoria2.extxyz")
        sys.exit(1)

    try:
        max_force = float(sys.argv[1])
    except ValueError:
        print("Error: El umbral de fuerza debe ser un número")
        sys.exit(1)

    input_files = sys.argv[2:]

    for input_file in input_files:
        if not os.path.isfile(input_file):
            print(f"Error: El archivo {input_file} no existe")
            continue
        
        print(f"\nProcesando archivo: {input_file}")
        # Ejecutar el filtrado
        output_file = filter_extxyz_by_forces(input_file, max_force)

        print(f"\nArchivo filtrado guardado como: {output_file}")

if __name__ == "__main__":
    main()
