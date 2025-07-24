#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ase.io import read
import matplotlib.pyplot as plt
import argparse
import numpy as np
from ase.data import atomic_numbers

# Parsear argumentos de línea de comandos
parser = argparse.ArgumentParser(description='Graficar propiedades desde un archivo .xyz y mostrar estadísticas')
parser.add_argument('xyz_file', type=str, help='Ruta al archivo .xyz')
args = parser.parse_args()

# Ruta del archivo .xyz
xyz_file = args.xyz_file

# Leer el archivo .xyz
data = read(xyz_file, index=':')

# Función para extraer elementos únicos (similar a get_elements_from_xyz.py)
def extraer_elementos_xyz(estructuras):
    """
    Extrae elementos únicos de las estructuras XYZ usando ASE.
    Cuenta cada elemento una sola vez por estructura.
    """
    elementos_por_estructura = {}
    
    for estructura in estructuras:
        # Obtener elementos únicos en esta estructura
        elementos_estructura = set(estructura.get_chemical_symbols())
        
        # Contar cada elemento único una vez por estructura
        for elemento in elementos_estructura:
            elementos_por_estructura[elemento] = elementos_por_estructura.get(elemento, 0) + 1
    
    return elementos_por_estructura

# Función para análisis de número de átomos (similar a numatoms.py)
def analizar_num_atomos(estructuras):
    """
    Analiza estadísticas del número de átomos por estructura.
    """
    num_atomos = [len(atoms) for atoms in estructuras]
    media = np.mean(num_atomos)
    minimo = np.min(num_atomos)
    maximo = np.max(num_atomos)
    
    return num_atomos, media, minimo, maximo

# Análisis de elementos únicos
elementos_por_estructura = extraer_elementos_xyz(data)
total_estructuras = len(data)

# Análisis de número de átomos
num_atoms, media_atomos, min_atomos, max_atomos = analizar_num_atomos(data)

# Mostrar estadísticas de elementos
print(f"\n📊 ANÁLISIS DE ELEMENTOS")
print(f"Total de estructuras encontradas: {total_estructuras}")
print("\nElementos encontrados con sus porcentajes:")
print("=" * 50)

elementos_ordenados = sorted(elementos_por_estructura.keys())
for elemento in elementos_ordenados:
    count = elementos_por_estructura[elemento]
    porcentaje = (count / total_estructuras) * 100
    numero_atomico = atomic_numbers.get(elemento, 0)
    print(f"{elemento} (Z={numero_atomico}): {count}/{total_estructuras} estructuras ({porcentaje:.1f}%)")

print("\n" + "=" * 50)
print(f"Elementos únicos encontrados: {elementos_ordenados}")
print(f"Total de elementos diferentes: {len(elementos_ordenados)}")

# Mostrar estadísticas de número de átomos
print(f"\n📊 ANÁLISIS DE NÚMERO DE ÁTOMOS")
print(f"Archivo analizado: {xyz_file}")
print(f"Número total de estructuras: {total_estructuras}")
print(f"Número medio de átomos por estructura: {media_atomos:.2f}")
print(f"Número mínimo de átomos en una estructura: {min_atomos}")
print(f"Número máximo de átomos en una estructura: {max_atomos}")
print("=" * 50)

# Extraer el número de átomos y la energía por átomo
energy_per_atom = [atoms.get_potential_energy() / len(atoms) for atoms in data]

# Extraer las fuerzas y los stresses totales
forces = [np.linalg.norm(atoms.get_forces()) for atoms in data]
stresses = [np.linalg.norm(atoms.get_stress()) for atoms in data]

# Crear una figura con 6 subfiguras (3 filas, 2 columnas)
fig, axs = plt.subplots(3, 2, figsize=(12,8))

# Fila 1: Energía por átomo
axs[0,0].scatter(energy_per_atom, energy_per_atom)
axs[0,0].plot(energy_per_atom, energy_per_atom, color='black', linestyle='--', label='x = y')
axs[0,0].set_xlabel('Energía por Átomo (eV)')
axs[0,0].set_ylabel('Energía por Átomo (eV)')
axs[0,0].legend()

axs[0,1].hist(energy_per_atom, bins=20, color='blue', alpha=0.7, edgecolor='black')
axs[0,1].set_xlabel('Energía por Átomo (eV)')
axs[0,1].set_ylabel('Frecuencia')

# Fila 2: Fuerzas
axs[1,0].scatter(forces, forces)
axs[1,0].plot(forces, forces, color='black', linestyle='--', label='x = y')
axs[1,0].set_xlabel('Fuerzas Totales (eV/Å)')
axs[1,0].set_ylabel('Fuerzas Totales (eV/Å)')
axs[1,0].legend()

axs[1,1].hist(forces, bins=20, color='green', alpha=0.7, edgecolor='black')
axs[1,1].set_xlabel('Fuerzas Totales (eV/Å)')
axs[1,1].set_ylabel('Frecuencia')

# Fila 3: Stresses
axs[2,0].scatter(stresses, stresses)
axs[2,0].plot(stresses, stresses, color='black', linestyle='--', label='x = y')
axs[2,0].set_xlabel('Stresses Totales (eV/Å³)')
axs[2,0].set_ylabel('Stresses Totales (eV/Å³)')
axs[2,0].legend()

axs[2,1].hist(stresses, bins=20, color='orange', alpha=0.7, edgecolor='black')
axs[2,1].set_xlabel('Stresses Totales (eV/Å³)')
axs[2,1].set_ylabel('Frecuencia')

# Ajustar el espaciado entre subgráficos
plt.tight_layout()
plt.show()

# Mostrar información adicional de elementos como texto
print(f"\n📈 RESUMEN DE GRÁFICOS GENERADOS:")
print(f"- Energía por átomo: {len(energy_per_atom)} puntos")
print(f"- Energía por átomo: rango [{min(energy_per_atom):.3f}, {max(energy_per_atom):.3f}] eV")
print(f"- Fuerzas totales: rango [{min(forces):.3f}, {max(forces):.3f}] eV/Å")
print(f"- Stresses totales: rango [{min(stresses):.3f}, {max(stresses):.3f}] eV/Å³")
print(f"- Número de átomos: rango [{min_atomos}, {max_atomos}] átomos")