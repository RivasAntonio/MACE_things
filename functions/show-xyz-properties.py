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
parser.add_argument('--energy-min', type=float, default=None, help='Filtrar: energía total mínima (eV)')
parser.add_argument('--energy-max', type=float, default=None, help='Filtrar: energía total máxima (eV)')
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
    print(f"{elemento} (Z={numero_atomico}): {count}/{total_estructuras} estructuras ({porcentaje:.4f}%)")

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

# Función para detectar el tipo de etiquetas en el archivo
def detectar_tipo_etiquetas(estructuras):
    """
    Detecta si el archivo usa etiquetas ASE estándar o REF_.
    Retorna 'ASE' o 'REF' según el tipo encontrado.
    """
    if len(estructuras) > 0:
        primera_estructura = estructuras[0]
        
        # Verificar si tiene etiquetas REF_
        tiene_ref_energy = 'REF_energy' in primera_estructura.info
        tiene_ref_forces = 'REF_forces' in primera_estructura.arrays
        tiene_ref_stress = 'REF_stress' in primera_estructura.info
        
        if tiene_ref_energy or tiene_ref_forces or tiene_ref_stress:
            return 'REF'
    return 'ASE'  # Por defecto ASE

# Función para obtener propiedades según el tipo de etiquetas
def obtener_propiedad(atoms, propiedad, tipo_etiquetas):
    """
    Obtiene una propiedad de manera uniforme según el tipo de etiquetas.
    Para ASE usa los métodos estándar get_*, para REF usa info/arrays con prefijo REF_.
    """
    try:
        if tipo_etiquetas == 'ASE':
            # Para ASE, usar métodos estándar
            if propiedad == 'energy':
                return atoms.get_potential_energy()
            elif propiedad == 'forces':
                return atoms.get_forces()
            elif propiedad == 'stress':
                return atoms.get_stress()
        else:  # REF
            # Para REF, usar info y arrays con prefijo REF_
            if propiedad == 'energy':
                return atoms.info.get('REF_energy')
            elif propiedad == 'forces':
                return atoms.arrays.get('REF_forces')
            elif propiedad == 'stress':
                return atoms.info.get('REF_stress')
    except:
        pass
    return None

# Detectar tipo de etiquetas
tipo_etiquetas = detectar_tipo_etiquetas(data)

# Extraer propiedades según el tipo detectado

# Filtrar por energía total si se especifica
energy_per_atom = []
energy_total = []
forces = []
stresses = []
filtered_data = []

for atoms in data:
    energy = obtener_propiedad(atoms, 'energy', tipo_etiquetas)
    if energy is not None:
        # Aplicar filtro si corresponde
        if (args.energy_min is not None and energy < args.energy_min):
            continue
        if (args.energy_max is not None and energy > args.energy_max):
            continue
        filtered_data.append(atoms)
        energy_total.append(energy)
        energy_per_atom.append(energy / len(atoms))
        force = obtener_propiedad(atoms, 'forces', tipo_etiquetas)
        if force is not None:
            forces.append(np.max(np.linalg.norm(force, axis=1)))
        stress = obtener_propiedad(atoms, 'stress', tipo_etiquetas)
        if stress is not None:
            stresses.append(np.linalg.norm(stress))

# Usar filtered_data para análisis de elementos y número de átomos
if args.energy_min is not None or args.energy_max is not None:
    elementos_por_estructura = extraer_elementos_xyz(filtered_data)
    total_estructuras = len(filtered_data)
    num_atoms, media_atomos, min_atomos, max_atomos = analizar_num_atomos(filtered_data)


# Identificar puntos con config_type=IsolatedAtom
isolated_atom_indices = []
for i, atoms in enumerate(filtered_data):
    if atoms.info.get('config_type') == 'IsolatedAtom':
        isolated_atom_indices.append(i)

# Crear una figura con 8 subfiguras (2 filas, 4 columnas)
if len(energy_per_atom) > 0 or len(forces) > 0 or len(stresses) > 0 or len(energy_total) > 0:
    fig, axs = plt.subplots(2, 4, figsize=(16,8))

    # Energía total
    if len(energy_total) > 0:
        x_indices = np.array(range(len(energy_total))) / 1000  # Convertir a millares
        
        # Scatter plot normal
        axs[0,0].scatter(x_indices, energy_total, color='blue', alpha=0.6)
        
        # Marcar puntos de IsolatedAtom en rojo
        if isolated_atom_indices:
            isolated_x = np.array(isolated_atom_indices) / 1000
            isolated_y = [energy_total[i] for i in isolated_atom_indices]
            axs[0,0].scatter(isolated_x, isolated_y, color='red', alpha=0.8, label='IAE')
            axs[0,0].legend()
        
        axs[0,0].set_xlabel('Índice de estructura (×1000)')
        axs[0,0].set_ylabel('Energía total (eV)')
        axs[0,0].set_title('Energía total')

        axs[1,0].hist(energy_total, bins=20, color='purple', alpha=0.7, edgecolor='black')
        axs[1,0].set_xlabel('Energía total (eV)')
        axs[1,0].set_ylabel('Frecuencia')
        axs[1,0].set_title('Histograma energía total')
    else:
        axs[0,0].text(0.5, 0.5, 'No hay datos de energía total', ha='center', va='center', transform=axs[0,0].transAxes)
        axs[1,0].text(0.5, 0.5, 'No hay datos de energía total', ha='center', va='center', transform=axs[1,0].transAxes)

    # Energía por átomo
    if len(energy_per_atom) > 0:
        # Scatter plot normal
        axs[0,1].scatter(energy_per_atom, energy_per_atom, color='blue', alpha=0.6)
        axs[0,1].plot(energy_per_atom, energy_per_atom, color='black', linestyle='--', alpha=0.5)
        
        # Marcar puntos de IsolatedAtom en rojo
        if isolated_atom_indices:
            isolated_energy_per_atom = [energy_per_atom[i] for i in isolated_atom_indices]
            axs[0,1].scatter(isolated_energy_per_atom, isolated_energy_per_atom, color='red', alpha=0.8, label='IAE')
            axs[0,1].legend()
        
        axs[0,1].set_xlabel('Energía por Átomo (eV)')
        axs[0,1].set_ylabel('Energía por Átomo (eV)')

        axs[1,1].hist(energy_per_atom, bins=20, color='blue', alpha=0.7, edgecolor='black')
        axs[1,1].set_xlabel('Energía por Átomo (eV)')
        axs[1,1].set_ylabel('Frecuencia')
    else:
        axs[0,1].text(0.5, 0.5, 'No hay datos de energía', ha='center', va='center', transform=axs[0,1].transAxes)
        axs[1,1].text(0.5, 0.5, 'No hay datos de energía', ha='center', va='center', transform=axs[1,1].transAxes)

    # Fuerzas
    if len(forces) > 0:
        # Scatter plot normal
        axs[0,2].scatter(forces, forces, color='blue', alpha=0.6)
        axs[0,2].plot(forces, forces, color='black', linestyle='--', alpha=0.5)
        
        # Marcar puntos de IsolatedAtom en rojo
        if isolated_atom_indices:
            isolated_forces = [forces[i] for i in isolated_atom_indices if i < len(forces)]
            if isolated_forces:
                axs[0,2].scatter(isolated_forces, isolated_forces, color='red', alpha=0.8, label='IAE')
                axs[0,2].legend()
        
        axs[0,2].set_xlabel('Máxima Fuerza por Átomo (eV/Å)')
        axs[0,2].set_ylabel('Máxima Fuerza por Átomo (eV/Å)')

        axs[1,2].hist(forces, bins=20, color='green', alpha=0.7, edgecolor='black')
        axs[1,2].set_xlabel('Máxima Fuerza por Átomo (eV/Å)')
        axs[1,2].set_ylabel('Frecuencia')
    else:
        axs[0,2].text(0.5, 0.5, 'No hay datos de fuerzas', ha='center', va='center', transform=axs[0,2].transAxes)
        axs[1,2].text(0.5, 0.5, 'No hay datos de fuerzas', ha='center', va='center', transform=axs[1,2].transAxes)

    # Stresses
    if len(stresses) > 0:
        # Scatter plot normal
        axs[0,3].scatter(stresses, stresses, color='blue', alpha=0.6)
        axs[0,3].plot(stresses, stresses, color='black', linestyle='--', alpha=0.5)
        
        # Marcar puntos de IsolatedAtom en rojo
        if isolated_atom_indices:
            isolated_stresses = [stresses[i] for i in isolated_atom_indices if i < len(stresses)]
            if isolated_stresses:
                axs[0,3].scatter(isolated_stresses, isolated_stresses, color='red', alpha=0.8, label='IAE')
                axs[0,3].legend()
        
        axs[0,3].set_xlabel('Stresses Totales (eV/Å³)')
        axs[0,3].set_ylabel('Stresses Totales (eV/Å³)')

        axs[1,3].hist(stresses, bins=20, color='orange', alpha=0.7, edgecolor='black')
        axs[1,3].set_xlabel('Stresses Totales (eV/Å³)')
        axs[1,3].set_ylabel('Frecuencia')
    else:
        axs[0,3].text(0.5, 0.5, 'No hay datos de stress', ha='center', va='center', transform=axs[0,3].transAxes)
        axs[1,3].text(0.5, 0.5, 'No hay datos de stress', ha='center', va='center', transform=axs[1,3].transAxes)

    # Ajustar el espaciado entre subgráficos
    plt.tight_layout()
    plt.show()
else:
    print("⚠️  No se encontraron datos de propiedades para graficar")

# Mostrar información adicional de elementos como texto
print(f"\n📈 RESUMEN DE GRÁFICOS GENERADOS:")
print(f"- Energía por átomo: {len(energy_per_atom)} puntos de {total_estructuras} estructuras")
if len(energy_per_atom) > 0:
    print(f"- Energía por átomo: rango [{min(energy_per_atom):.3f}, {max(energy_per_atom):.3f}] eV")
if len(forces) > 0:
    print(f"- Máxima fuerza por átomo: {len(forces)} puntos, rango [{min(forces):.3f}, {max(forces):.3f}] eV/Å")
if len(stresses) > 0:
    print(f"- Stresses totales: {len(stresses)} puntos, rango [{min(stresses):.3f}, {max(stresses):.3f}] eV/Å³")
print(f"- Número de átomos: rango [{min_atomos}, {max_atomos}] átomos")
