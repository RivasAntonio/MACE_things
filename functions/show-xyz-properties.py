#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect and plot key properties stored in XYZ and EXTXYZ structure files."""

from ase.io import read
import matplotlib.pyplot as plt
import argparse
import numpy as np
from ase.data import atomic_numbers

# Conversión: 1 eV/Å^3 = 160.21766208 GPa
EV_PER_A3_TO_GPA = 160.21766208
# Parsear argumentos de línea de comandos

parser = argparse.ArgumentParser(
    description='''
Analyze and visualize physicochemical properties from .xyz structure files

FEATURES:
• Element analysis: chemical composition and percentages
• Atomic statistics: mean, min, max atoms per structure
• Property visualization: energies, forces, and stresses
• Statistical histograms with mean and median lines
• Auto-detection: ASE standard or REF_ label formats
• Energy filtering for data subsets
• Isolated atom identification (marked as IAE in plots)

OUTPUT: Interactive plots (3×4 subplots) + detailed statistical summary
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument('xyz_file', type=str, help='Path to .xyz file containing structures to analyze')
parser.add_argument('--energy-min', type=float, default=None, help='Filter structures with minimum total energy (eV)')
parser.add_argument('--energy-max', type=float, default=None, help='Filter structures with maximum total energy (eV)')
parser.add_argument('--no-plots', action='store_true', help='Disable plot generation and display, only print summary')
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
max_force_components = []  # Máxima componente de fuerza por configuración
stress_xx = []
stress_yy = []
stress_zz = []
stress_yz = []
stress_xz = []
stress_xy = []
pressures = []  # Presión = 1/3 de la traza del tensor de stress
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
            # Almacenar la componente máxima en valor absoluto de cada configuración
            max_component = np.max(np.abs(force))
            # Mantener el signo de la componente máxima original
            flat_forces = force.flatten()
            max_idx = np.argmax(np.abs(flat_forces))
            max_force_components.append(flat_forces[max_idx])
        stress = obtener_propiedad(atoms, 'stress', tipo_etiquetas)
        if stress is not None:
            # Almacenar cada componente del stress individualmente
            # En formato Voigt de ASE: [xx, yy, zz, yz, xz, xy]
            stress_xx.append(stress[0])
            stress_yy.append(stress[1])
            stress_zz.append(stress[2])
            stress_yz.append(stress[3])
            stress_xz.append(stress[4])
            stress_xy.append(stress[5])
            # Calcular presión como 1/3 de la traza (componentes diagonales xx, yy, zz)
            pressure_ev_per_a3 = (stress[0] + stress[1] + stress[2]) / 3.0
            # Convertir a GPa
            pressure_gpa = pressure_ev_per_a3 * EV_PER_A3_TO_GPA
            pressures.append(pressure_gpa)

# Calcular número de átomos para filtered_data (para usar en gráficos)
num_atoms, media_atomos_filtered, min_atomos_filtered, max_atomos_filtered = analizar_num_atomos(filtered_data)

# Identificar puntos con config_type=IsolatedAtom
isolated_atom_indices = []
for i, atoms in enumerate(filtered_data):
    if atoms.info.get('config_type') == 'IsolatedAtom':
        isolated_atom_indices.append(i)

# Crear una figura con 12 subfiguras (3 filas, 4 columnas)
if not args.no_plots and (len(energy_per_atom) > 0 or len(max_force_components) > 0 or len(stress_xx) > 0 or len(energy_total) > 0):
    fig, axs = plt.subplots(3, 4, figsize=(15,10))

    # Energía por átomo con color por frecuencia
    if len(energy_per_atom) > 0:
        # Calcular histograma 2D para obtener frecuencias
        hist, xedges, yedges = np.histogram2d(energy_per_atom, energy_per_atom, bins=30)
        
        # Asignar cada punto a su bin y obtener la frecuencia
        x_bins = np.digitize(energy_per_atom, xedges) - 1
        y_bins = np.digitize(energy_per_atom, yedges) - 1
        # Asegurar que los índices estén dentro de los límites
        x_bins = np.clip(x_bins, 0, hist.shape[0] - 1)
        y_bins = np.clip(y_bins, 0, hist.shape[1] - 1)
        
        frequencies = hist[x_bins, y_bins]
        
        # Scatter plot con color por frecuencia
        scatter = axs[0,0].scatter(energy_per_atom, energy_per_atom, c=frequencies, cmap='inferno', s=20)
        axs[0,0].plot(energy_per_atom, energy_per_atom, color='black', linestyle='--', alpha=0.3)
        
        # Añadir barra de color
        cbar = plt.colorbar(scatter, ax=axs[0,0])
        cbar.set_label('Frecuencia')
        
        # Marcar puntos de IsolatedAtom en rojo
        if isolated_atom_indices:
            isolated_energy_per_atom = [energy_per_atom[i] for i in isolated_atom_indices]
            axs[0,0].scatter(isolated_energy_per_atom, isolated_energy_per_atom, color='red', s=30, label='IAE', zorder=5)
            axs[0,0].legend()
        
        axs[0,0].set_xlabel('Energía por Átomo (eV)')
        axs[0,0].set_ylabel('Energía por Átomo (eV)')
    else:
        axs[0,0].text(0.5, 0.5, 'No hay datos de energía', ha='center', va='center', transform=axs[0,0].transAxes)

    # Fila 2: Histograma de energía por átomo (primera columna, segunda fila)
    if len(energy_per_atom) > 0:
        axs[1,0].hist(energy_per_atom, bins=30, color='teal', edgecolor='black')
        # Añadir líneas de media y mediana
        mean_energy_pa = np.mean(energy_per_atom)
        median_energy_pa = np.median(energy_per_atom)
        axs[1,0].axvline(mean_energy_pa, color='red', linestyle='--', linewidth=2, label=f'$\\bar{{x}}$: {mean_energy_pa:.3f}')
        axs[1,0].axvline(median_energy_pa, color='orange', linestyle='--', linewidth=2, label=f'$\\mu$: {median_energy_pa:.3f}')
        axs[1,0].legend()
        axs[1,0].set_xlabel('Energía por átomo (eV)')
        axs[1,0].set_ylabel('Frecuencia')
        axs[1,0].set_title('Histograma de energía por átomo')
    else:
        axs[1,0].text(0.5, 0.5, 'No hay datos de energía por átomo', ha='center', va='center', transform=axs[1,0].transAxes)

    # Fila 3: Panel de estadísticas globales (primera columna, tercera fila)
    axs[2,0].axis('off')
    fig.canvas.manager.set_window_title(f"{args.xyz_file}")
    stats_text = f"ESTADÍSTICAS GLOBALES" + "\n" + "="*35 + "\n\n"
    
    # Estadísticas de energía por átomo (rango + percentil 95)
    if len(energy_per_atom) > 0:
        min_energy_pa = float(np.min(energy_per_atom))
        max_energy_pa = float(np.max(energy_per_atom))
        p95_energy_pa = float(np.percentile(energy_per_atom, 95))
        stats_text += f"Energía por átomo (eV):\n"
        stats_text += f"  Rango: [{min_energy_pa:>7.3f}, {max_energy_pa:>7.3f}]\n"
        stats_text += f"  Percentil 95: {p95_energy_pa:>10.3f}\n"
    
    # Estadísticas de fuerzas (rango + percentil 95)
    if len(max_force_components) > 0:
        min_forces = float(np.min(max_force_components))
        max_forces = float(np.max(max_force_components))
        p95_forces = float(np.percentile(max_force_components, 95))
        stats_text += f"Fuerzas (eV/Å):\n"
        stats_text += f"  Rango: [{min_forces:>7.3f}, {max_forces:>7.3f}]\n"
        stats_text += f"  Percentil 95: {p95_forces:>10.3f}\n"
    
    # Estadísticas de stresses (rango + percentil 95)
    if len(stress_xx) > 0:
        all_stress_components = np.concatenate([stress_xx, stress_yy, stress_zz, stress_yz, stress_xz, stress_xy])
        min_stresses = float(np.min(all_stress_components))
        max_stresses = float(np.max(all_stress_components))
        p95_stresses = float(np.percentile(all_stress_components, 95))
        stats_text += f"Stresses (eV/Å³):\n"
        stats_text += f"  Rango: [{min_stresses:>7.3f}, {max_stresses:>7.3f}]\n"
        stats_text += f"  Percentil 95: {p95_stresses:>10.3f}\n"
    
    # Estadísticas de presiones (rango + percentil 95)
    if len(pressures) > 0:
        min_pressures = float(np.min(pressures))
        max_pressures = float(np.max(pressures))
        p95_pressures = float(np.percentile(pressures, 95))
        stats_text += f"Presiones (GPa):\n"
        stats_text += f"  Rango: [{min_pressures:>7.3f}, {max_pressures:>7.3f}]\n"
        stats_text += f"  Percentil 95: {p95_pressures:>10.3f}\n"
    
    axs[2,0].text(0.1, 0.95, stats_text, ha='left', va='top', transform=axs[2,0].transAxes,
                  fontfamily='monospace', fontsize=9,
                  bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', alpha=0.8))

    # Fuerzas
    if len(max_force_components) > 0:
        # Calcular la norma máxima por estructura para el scatter plot
        force_norms = []
        for atoms in filtered_data:
            force = obtener_propiedad(atoms, 'forces', tipo_etiquetas)
            if force is not None:
                force_norms.append(np.max(np.linalg.norm(force, axis=1)))
        
        # Calcular histograma 2D para obtener frecuencias
        hist_f, xedges_f, yedges_f = np.histogram2d(force_norms, force_norms, bins=30)
        x_bins_f = np.digitize(force_norms, xedges_f) - 1
        y_bins_f = np.digitize(force_norms, yedges_f) - 1
        x_bins_f = np.clip(x_bins_f, 0, hist_f.shape[0] - 1)
        y_bins_f = np.clip(y_bins_f, 0, hist_f.shape[1] - 1)
        frequencies_f = hist_f[x_bins_f, y_bins_f]
        
        # Scatter plot con color por frecuencia
        scatter_f = axs[0,1].scatter(force_norms, force_norms, c=frequencies_f, cmap='inferno', s=20)
        axs[0,1].plot(force_norms, force_norms, color='black', linestyle='--', alpha=0.3)
        
        # Añadir barra de color
        cbar_f = plt.colorbar(scatter_f, ax=axs[0,1])
        cbar_f.set_label('Frecuencia')
        
        # Marcar puntos de IsolatedAtom en rojo
        if isolated_atom_indices:
            isolated_forces = [force_norms[i] for i in isolated_atom_indices if i < len(force_norms)]
            if isolated_forces:
                axs[0,1].scatter(isolated_forces, isolated_forces, color='red', s=30, label='IAE', zorder=5)
                axs[0,1].legend()
        
        axs[0,1].set_xlabel('Máxima Fuerza por Átomo (eV/Å)')
        axs[0,1].set_ylabel('Máxima Fuerza por Átomo (eV/Å)')

        # Fila 2: Histograma de la máxima componente de fuerza por configuración
        axs[1,1].hist(max_force_components, bins=30, color='green', edgecolor='black')
        mean_forces = np.mean(max_force_components)
        median_forces = np.median(max_force_components)
        axs[1,1].axvline(mean_forces, color='red', linestyle='--', linewidth=2, label=f'$\\bar{{x}}$: {mean_forces:.3f}')
        axs[1,1].axvline(median_forces, color='blue', linestyle='--', linewidth=2, label=f'$\\mu$: {median_forces:.3f}')
        axs[1,1].legend()
        axs[1,1].set_xlabel('Máxima Componente de Fuerza (eV/Å)')
        axs[1,1].set_ylabel('Frecuencia')
        axs[1,1].set_yscale('log')
        axs[1,1].set_title('Histograma de fuerzas')

        # Fila 3: Distribución log-log para fuerzas (segunda columna, tercera fila)
        counts, bins = np.histogram(force_norms, bins=30, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        axs[2,1].scatter(bin_centers, counts, color='green',  s=50,alpha=0.25+0.75*counts/max(counts))
        # Añadir una ley de potencias con exponente -5/2 (línea discontinua)
        try:
            # Evitar ceros y calcular un rango log-spaced para la línea
            positive = bin_centers > 0
            if np.any(positive) and np.any(counts[positive] > 0):
                x_min = bin_centers[positive].min()
                x_max = bin_centers[positive].max()
                x_line = np.logspace(np.log10(x_min), np.log10(x_max), 200)
                # Escalar la ley de potencias para que pase cerca del pico del histograma
                peak_idx = np.argmax(counts)
                peak_x = bin_centers[peak_idx] if bin_centers[peak_idx] > 0 else x_min
                C = counts[peak_idx] * (peak_x ** 2)
                y_line = C * x_line ** -(5/2)
                axs[2,1].plot(x_line, y_line, color='black', linestyle='--', linewidth=1.5, label=r'$\propto x^{-5/2}$')
                axs[2,1].legend()
        except Exception:
            pass
        axs[2,1].set_xlabel('Máxima Fuerza por Átomo (eV/Å)')
        axs[2,1].set_ylabel('Densidad de probabilidad')
        axs[2,1].set_xscale('log')
        axs[2,1].set_yscale('log')

    else:
        axs[0,1].text(0.5, 0.5, 'No hay datos de fuerzas', ha='center', va='center', transform=axs[0,1].transAxes)
        axs[1,1].text(0.5, 0.5, 'No hay datos de fuerzas', ha='center', va='center', transform=axs[1,1].transAxes)
        axs[2,1].text(0.5, 0.5, 'No hay datos de fuerzas', ha='center', va='center', transform=axs[2,1].transAxes)

    # Stresses - Todas las componentes
    if len(stress_xx) > 0:
        # Fila 1: Scatter plot - mantener la norma para visualización
        stress_norms = [np.linalg.norm([stress_xx[i], stress_yy[i], stress_zz[i], stress_yz[i], stress_xz[i], stress_xy[i]]) 
                        for i in range(len(stress_xx))]
        
        # Calcular histograma 2D para obtener frecuencias
        hist_s, xedges_s, yedges_s = np.histogram2d(stress_norms, stress_norms, bins=30)
        x_bins_s = np.digitize(stress_norms, xedges_s) - 1
        y_bins_s = np.digitize(stress_norms, yedges_s) - 1
        x_bins_s = np.clip(x_bins_s, 0, hist_s.shape[0] - 1)
        y_bins_s = np.clip(y_bins_s, 0, hist_s.shape[1] - 1)
        frequencies_s = hist_s[x_bins_s, y_bins_s]
        
        # Scatter plot con color por frecuencia
        scatter_s = axs[0,2].scatter(stress_norms, stress_norms, c=frequencies_s, cmap='inferno', s=20)
        axs[0,2].plot(stress_norms, stress_norms, color='black', linestyle='--', alpha=0.3)
        
        # Añadir barra de color
        cbar_s = plt.colorbar(scatter_s, ax=axs[0,2])
        cbar_s.set_label('Frecuencia')
        
        # Marcar puntos de IsolatedAtom en rojo
        if isolated_atom_indices:
            isolated_stresses = [stress_norms[i] for i in isolated_atom_indices if i < len(stress_norms)]
            if isolated_stresses:
                axs[0,2].scatter(isolated_stresses, isolated_stresses, color='red', s=30, label='IAE', zorder=5)
                axs[0,2].legend()
        
        axs[0,2].set_xlabel('||Stress|| (eV/Å³)')
        axs[0,2].set_ylabel('||Stress|| (eV/Å³)')

        # Fila 2: Histograma de todas las componentes del stress juntas
        all_stress_components = np.concatenate([stress_xx, stress_yy, stress_zz, stress_yz, stress_xz, stress_xy])
        axs[1,2].hist(all_stress_components, bins=30, color='orange', edgecolor='black')
        mean_all_stresses = np.mean(all_stress_components)
        median_all_stresses = np.median(all_stress_components)
        axs[1,2].axvline(mean_all_stresses, color='red', linestyle='--', linewidth=2, label=f'$\\bar{{x}}$: {mean_all_stresses:.3f}')
        axs[1,2].axvline(median_all_stresses, color='blue', linestyle='--', linewidth=2, label=f'$\\mu$: {median_all_stresses:.3f}')
        axs[1,2].legend()
        axs[1,2].set_yscale('log')
        axs[1,2].set_xlabel('Componentes de Stress (eV/Å³)')
        axs[1,2].set_ylabel('Frecuencia')
        axs[1,2].set_title('Histograma de stresses')

        # Fila 3: Distribución log-log para la norma del stress
        counts, bins = np.histogram(stress_norms, bins=30, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        axs[2,2].scatter(bin_centers, counts, color='orange',  s=50, alpha=0.25+0.75*counts/max(counts))
        # Añadir una ley de potencias con exponente -2 (línea discontinua)
        try:
            positive = bin_centers > 0
            if np.any(positive) and np.any(counts[positive] > 0):
                x_min = bin_centers[positive].min()
                x_max = bin_centers[positive].max()
                x_line = np.logspace(np.log10(x_min), np.log10(x_max), 200)
                peak_idx = np.argmax(counts)
                peak_x = bin_centers[peak_idx] if bin_centers[peak_idx] > 0 else x_min
                C = counts[peak_idx] * (peak_x ** 2)
                y_line = C * x_line ** -2
                axs[2,2].plot(x_line, y_line, color='black', linestyle='--', linewidth=1.5, label=r'$\propto x^{-2}$')
                axs[2,2].legend()
        except Exception:
            pass
        
        axs[2,2].set_xlabel('||Stress|| (eV/Å³)')
        axs[2,2].set_ylabel('Densidad de probabilidad')
        axs[2,2].set_xscale('log')
        axs[2,2].set_yscale('log')

    else:
        axs[0,2].text(0.5, 0.5, 'No hay datos de stress', ha='center', va='center', transform=axs[0,2].transAxes)
        axs[1,2].text(0.5, 0.5, 'No hay datos de stress', ha='center', va='center', transform=axs[1,2].transAxes)
        axs[2,2].text(0.5, 0.5, 'No hay datos de stress', ha='center', va='center', transform=axs[2,2].transAxes)

    # Presiones (columna 4)
    if len(pressures) > 0:
        # Calcular histograma 2D para obtener frecuencias
        hist_p, xedges_p, yedges_p = np.histogram2d(pressures, pressures, bins=30)
        x_bins_p = np.digitize(pressures, xedges_p) - 1
        y_bins_p = np.digitize(pressures, yedges_p) - 1
        x_bins_p = np.clip(x_bins_p, 0, hist_p.shape[0] - 1)
        y_bins_p = np.clip(y_bins_p, 0, hist_p.shape[1] - 1)
        frequencies_p = hist_p[x_bins_p, y_bins_p]
        
        # Scatter plot con color por frecuencia
        scatter_p = axs[0,3].scatter(pressures, pressures, c=frequencies_p, cmap='inferno', s=20)
        axs[0,3].plot(pressures, pressures, color='black', linestyle='--', alpha=0.3)
        
        # Añadir barra de color
        cbar_p = plt.colorbar(scatter_p, ax=axs[0,3])
        cbar_p.set_label('Frecuencia')
        
        # Marcar puntos de IsolatedAtom en rojo
        if isolated_atom_indices:
            isolated_pressures = [pressures[i] for i in isolated_atom_indices if i < len(pressures)]
            if isolated_pressures:
                axs[0,3].scatter(isolated_pressures, isolated_pressures, color='red', s=30, label='IAE', zorder=5)
                axs[0,3].legend()
        
        axs[0,3].set_xlabel('Presión (GPa)')
        axs[0,3].set_ylabel('Presión (GPa)')

        # Fila 2: Histograma de presiones (cuarta columna, segunda fila)
        axs[1,3].hist(pressures, bins=30, color='purple', edgecolor='black')
        mean_pressures = np.mean(pressures)
        median_pressures = np.median(pressures)
        axs[1,3].axvline(mean_pressures, color='red', linestyle='--', linewidth=2, label=f'$\\bar{{x}}$: {mean_pressures:.3f}')
        axs[1,3].axvline(median_pressures, color='orange', linestyle='--', linewidth=2, label=f'$\\mu$: {median_pressures:.3f}')
        axs[1,3].legend()
        axs[1,3].set_yscale('log')
        axs[1,3].set_xlabel('Presión (GPa)')
        axs[1,3].set_ylabel('Frecuencia')
        axs[1,3].set_title('Histograma de presiones')

        # Fila 3: Distribución log-log para presiones (cuarta columna, tercera fila) - usar valores absolutos para escala log
        pressures_abs = np.abs(pressures)
        counts_p, bins_p = np.histogram(pressures_abs, bins=30, density=True)
        bin_centers_p = (bins_p[:-1] + bins_p[1:]) / 2
        axs[2,3].scatter(bin_centers_p, counts_p, color='purple', s=50, alpha=0.25+0.75*counts_p/max(counts_p))
        # Añadir una ley de potencias con exponente -2 (línea discontinua)
        try:
            positive_p = bin_centers_p > 0
            if np.any(positive_p) and np.any(counts_p[positive_p] > 0):
                x_min_p = bin_centers_p[positive_p].min()
                x_max_p = bin_centers_p[positive_p].max()
                x_line_p = np.logspace(np.log10(x_min_p), np.log10(x_max_p), 200)
                peak_idx_p = np.argmax(counts_p)
                peak_x_p = bin_centers_p[peak_idx_p] if bin_centers_p[peak_idx_p] > 0 else x_min_p
                C_p = counts_p[peak_idx_p] * (peak_x_p ** 2)
                y_line_p = C_p * x_line_p ** -2
                axs[2,3].plot(x_line_p, y_line_p, color='black', linestyle='--', linewidth=1.5, label=r'$\propto x^{-2}$')
                axs[2,3].legend()
        except Exception:
            pass
        
        axs[2,3].set_xlabel('|Presión| (GPa)')
        axs[2,3].set_ylabel('Densidad de probabilidad')
        axs[2,3].set_xscale('log')
        axs[2,3].set_yscale('log')

    else:
        axs[0,3].text(0.5, 0.5, 'No hay datos de presión', ha='center', va='center', transform=axs[0,3].transAxes)
        axs[1,3].text(0.5, 0.5, 'No hay datos de presión', ha='center', va='center', transform=axs[1,3].transAxes)
        axs[2,3].text(0.5, 0.5, 'No hay datos de presión', ha='center', va='center', transform=axs[2,3].transAxes)

    # Ajustar el espaciado entre subgráficos
    plt.tight_layout()
    plt.show()
elif not args.no_plots:
    print("⚠️  No se encontraron datos de propiedades para graficar")

# Mostrar información adicional de elementos como texto

# Análisis de elementos únicos (sobre data completo, no filtered)
elementos_por_estructura = extraer_elementos_xyz(data)
total_estructuras = len(data)

# Análisis de número de átomos (sobre data completo, no filtered)
num_atoms_all, media_atomos, min_atomos, max_atomos = analizar_num_atomos(data)

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
print(f"\n📈 RESUMEN DE GRÁFICOS GENERADOS:")
if len(energy_total) > 0:
    mean_energy_total = np.mean(energy_total)
    median_energy_total = np.median(energy_total)
    print(f"- Energía total: rango [{min(energy_total):.3f}, {max(energy_total):.3f}] eV")
    print(f"  • Media: {mean_energy_total:.3f} eV, Mediana: {median_energy_total:.3f} eV")
if len(energy_per_atom) > 0:
    mean_energy_per_atom = np.mean(energy_per_atom)
    median_energy_per_atom = np.median(energy_per_atom)
    print(f"- Energía por átomo: rango [{min(energy_per_atom):.3f}, {max(energy_per_atom):.3f}] eV")
    print(f"  • Media: {mean_energy_per_atom:.3f} eV, Mediana: {median_energy_per_atom:.3f} eV")
if len(max_force_components) > 0:
    mean_forces = np.mean(max_force_components)
    median_forces = np.median(max_force_components)
    print(f"- Máxima componente de fuerza por configuración: {len(max_force_components)} valores, rango [{min(max_force_components):.3f}, {max(max_force_components):.3f}] eV/Å")
    print(f"  • Media: {mean_forces:.3f} eV/Å, Mediana: {median_forces:.3f} eV/Å")
if len(stress_xx) > 0:
    all_stress_components = np.concatenate([stress_xx, stress_yy, stress_zz, stress_yz, stress_xz, stress_xy])
    mean_stresses = np.mean(all_stress_components)
    median_stresses = np.median(all_stress_components)
    print(f"- Componentes de stress: {len(stress_xx)} puntos, rango [{min(all_stress_components):.3f}, {max(all_stress_components):.3f}] eV/Å³")
    print(f"  • Media: {mean_stresses:.3f} eV/Å³, Mediana: {median_stresses:.3f} eV/Å³")
if len(pressures) > 0:
    mean_pressures = np.mean(pressures)
    median_pressures = np.median(pressures)
    print(f"- Presión (1/3 traza): {len(pressures)} puntos, rango [{min(pressures):.3f}, {max(pressures):.3f}] GPa")
    print(f"  • Media: {mean_pressures:.3f} GPa, Mediana: {median_pressures:.3f} GPa")
print(f"- Número de átomos: rango [{min_atomos}, {max_atomos}] átomos")
