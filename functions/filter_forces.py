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
    output_file = f"{base_name}.xyz"
    
    # Contadores para estadísticas
    total_frames = 0
    kept_frames = 0
    
    # Leer el archivo frame por frame (eficiente en memoria)
    with open(output_file, 'w') as f_out:
        for atoms in read(input_file, index=':', format='extxyz'):
            total_frames += 1
            
            try:
                forces = atoms.get_forces()
            except AttributeError:
                print(f"Advertencia: Frame {total_frames} no contiene fuerzas válidas. Descartado.")
                continue

            # Calcular norma de las fuerzas para cada átomo
            force_magnitudes = np.linalg.norm(forces, axis=1)
            max_force = np.max(force_magnitudes)
            
            # Escribir solo si pasa el filtro
            if max_force <= max_force_threshold:
                if kept_frames == 0:
                    # Primer frame - escribir con formato completo
                    write(f_out, atoms, format='extxyz')
                else:
                    # Frames subsiguientes - modo append
                    write(f_out, atoms, format='extxyz', append=True)
                kept_frames += 1
            else:
                print(f"Descartado frame {total_frames}: Fuerza máxima = {max_force:.2f} eV/Å")
    
    # Reporte final
    print("\n" + "="*50)
    print(f"Proceso completado:")
    print(f"Archivo de entrada: {input_file}")
    print(f"Archivo filtrado: {output_file}")
    print(f"Umbral de fuerza: {max_force_threshold} eV/Å")
    print(f"Frames totales: {total_frames}")
    print(f"Frames conservados: {kept_frames} ({kept_frames/total_frames:.1%})")
    print(f"Frames descartados: {total_frames - kept_frames}")
    print("="*50)
    
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