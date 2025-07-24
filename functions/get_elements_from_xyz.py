#!/usr/bin/env python3
import sys
from ase.io import read
from ase.data import atomic_numbers


def extraer_elementos_xyz(ruta_archivo):
    """
    Extrae elementos únicos de un archivo XYZ usando ASE.
    Cuenta cada elemento una sola vez por estructura.
    """
    elementos_por_estructura = {}  # Diccionario para contar en cuántas estructuras aparece cada elemento
    
    try:
        # Leer todas las estructuras del archivo XYZ
        estructuras = read(ruta_archivo, index=':')
        
        # Si solo hay una estructura, convertirla a lista
        if not isinstance(estructuras, list):
            estructuras = [estructuras]
        
        total_estructuras = len(estructuras)
        
        # Procesar cada estructura
        for estructura in estructuras:
            # Obtener elementos únicos en esta estructura
            elementos_estructura = set(estructura.get_chemical_symbols())
            
            # Contar cada elemento único una vez por estructura
            for elemento in elementos_estructura:
                elementos_por_estructura[elemento] = elementos_por_estructura.get(elemento, 0) + 1
        
        return elementos_por_estructura, total_estructuras
    
    except Exception as e:
        print(f"Error al leer el archivo {ruta_archivo}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python get_elements_from_xyz.py archivo.xyz")
        sys.exit(1)

    archivo = sys.argv[1]
    elementos_por_estructura, total_estructuras = extraer_elementos_xyz(archivo)
    
    if total_estructuras == 0:
        print("No se encontraron estructuras en el archivo.")
        sys.exit(1)
    
    print(f"Total de estructuras encontradas: {total_estructuras}")
    print("\nElementos encontrados con sus porcentajes:")
    print("=" * 50)
    
    # Ordenar elementos alfabéticamente
    elementos_ordenados = sorted(elementos_por_estructura.keys())
    
    for elemento in elementos_ordenados:
        count = elementos_por_estructura[elemento]
        porcentaje = (count / total_estructuras) * 100
        numero_atomico = atomic_numbers.get(elemento, 0)  # 0 si no se encuentra
        print(f"{elemento} (Z={numero_atomico}): {count}/{total_estructuras} estructuras ({porcentaje:.1f}%)")
    
    # Resumen adicional
    print("\n" + "=" * 50)
    print(f"Elementos únicos encontrados: {elementos_ordenados}")
    print(f"Total de elementos diferentes: {len(elementos_ordenados)}")
