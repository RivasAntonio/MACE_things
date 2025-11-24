#!/usr/bin/env python3
"""
Script para calcular la expansión térmica de FAU mediante dinámica molecular NPT
Rango de temperaturas: 0 - 1200 K

Protocolo:
  1. Minimización inicial a T=0K y presión especificada
  2. Para cada temperatura:
     - Equilibración Langevin (volumen fijo, termostato estocástico)
     - Producción NPT (barostato + termostato)
"""

from ase.io import read, write
from ase.optimize import BFGS
from ase.filters import UnitCellFilter
from ase.md.nose_hoover_chain import MTKNPT
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.io.trajectory import Trajectory
from mace.calculators import MACECalculator
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cuequivariance_torch")

# ============================================================================
# ARGUMENTOS DE LÍNEA DE COMANDOS
# ============================================================================

parser = argparse.ArgumentParser(
    description='Cálculo de expansión térmica de FAU mediante dinámica molecular NPT',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--plot', '-p', action='store_true', default=False,
                    help='Generar gráficas de visualización')
parser.add_argument('--save-structures', '-s', action='store_true', default=False,
                    help='Guardar estructuras finales de cada temperatura')
parser.add_argument('--analyze', '-a', action='store_true', default=False,
                    help='Realizar análisis y ajuste lineal de expansión térmica')
args = parser.parse_args()

SAVE_PLOTS = args.plot
SAVE_FINAL_STRUCTURES = args.save_structures
DO_ANALYSIS = args.analyze

# ============================================================================
# PARÁMETROS DE SIMULACIÓN
# ============================================================================

model_path = "../../zeolite-mh-finetuning-source.model"
input_structure = "../structures/CONTCAR_FAU.vasp"

# Temperaturas a simular (K)
T_min = 298
T_max = 1273
temperatures = np.arange(T_min, T_max + 1, 150)

# Parámetros MD
pressure_GPa = 0.0  # Presión en GPa
timestep_fs = 0.5  # Paso de tiempo en fs
equilibration_ps = 50.0  # Tiempo de equilibración (NVT)
production_ps = 500.0  # Tiempo de producción (NPT)
dump_interval = 100  # Guardar cada N pasos

tdamp = 100 * timestep_fs
pdamp = 1000 * timestep_fs

equilibration_steps = int(equilibration_ps * 1000 / timestep_fs)
production_steps = int(production_ps * 1000 / timestep_fs)

# Crear directorio de outputs
output_dir = "outputs_thermal_expansion_good_thermobarostat"
os.makedirs(output_dir, exist_ok=True)
print(f"📁 Directorio de salida: {output_dir}\n")

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

print("="*70)
print(" EXPANSIÓN TÉRMICA FAU - DINÁMICA MOLECULAR NPT")
print("="*70)
print(f"\nEstructura inicial: {input_structure}")
print(f"Modelo: {model_path}")
print(f"\nParámetros de barrido:")
print(f"  Temperaturas: {T_min} - {T_max} K ({len(temperatures)} puntos)")
print(f"  Presión: {pressure_GPa} GPa")
print(f"\nParámetros MD:")
print(f"  Timestep: {timestep_fs} fs")
print(f"  Equilibración: {equilibration_ps} ps")
print(f"  Producción: {production_ps} ps")
print(f"  Tiempo total por T: {equilibration_ps + production_ps} ps")
print(f"\nParámetros termostato/barostato:")
print(f"  tdamp: {tdamp} fs")
print(f"  pdamp: {pdamp} fs")
print(f"\nCuEq: Activado")
print(f"Device: CUDA")
print(f"\nOpciones:")
print(f"  Guardar gráficas: {'SÍ' if SAVE_PLOTS else 'NO (usa --plot para activar)'}")
print(f"  Guardar estructuras finales: {'SÍ' if SAVE_FINAL_STRUCTURES else 'NO (usa --save-structures para activar)'}")
print(f"  Análisis y ajuste lineal: {'SÍ' if DO_ANALYSIS else 'NO (usa --analyze para activar)'}")
print("="*70 + "\n")

# Inicializar calculador MACE
calc = MACECalculator(
    model_paths=model_path,
    device="cuda",
    default_dtype="float64", # Cambiado a float64, originalmente era float64
     enable_cueq=True
)

# ============================================================================
# PASO 1: MINIMIZACIÓN INICIAL
# ============================================================================

print("🔄 PASO 1: MINIMIZACIÓN INICIAL A T=0K")
print("-"*70)

atoms_initial = read(input_structure)
atoms_initial.calc = calc

print(f"Estructura inicial:")
print(f"  Átomos: {len(atoms_initial)}")
print(f"  Energía inicial: {atoms_initial.get_potential_energy():.6f} eV")
print(f"  Volumen inicial: {atoms_initial.get_volume():.4f} Å³")

# Minimizar a la presión especificada
ucf = UnitCellFilter(atoms_initial, scalar_pressure=0.0)

opt = BFGS(ucf, logfile=f"{output_dir}/initial_minimization.log", trajectory=f"{output_dir}/initial_minimization.traj")
print(f"\nMinimizando a P = {pressure_GPa} GPa...")
opt.run(fmax=0.01)

energy_min = atoms_initial.get_potential_energy()
volume_min = atoms_initial.get_volume()

print(f"\n✓ Minimización completada:")
print(f"  Energía final: {energy_min:.6f} eV")
print(f"  Volumen final: {volume_min:.4f} Å³")

# Guardar estructura minimizada
minimized_file = f"{output_dir}/FAU_minimized_P{pressure_GPa}GPa.vasp"
write(minimized_file, atoms_initial)
print(f"💾 Estructura guardada: {minimized_file}\n")

# ============================================================================
# PASO 2: BARRIDO DE TEMPERATURAS
# ============================================================================

print("🔄 PASO 2: BARRIDO DE TEMPERATURAS")
print("-"*70)

# Arrays para almacenar resultados
volumes_mean = []
volumes_std = []
cell_a_mean = []
cell_b_mean = []
cell_c_mean = []
energies_mean = []
energies_std = []

# Variable para guardar la última estructura
last_atoms = None

for i, T in enumerate(temperatures):
    
    print(f"\n{'='*70}")
    print(f" TEMPERATURA {i+1}/{len(temperatures)}: {T:.1f} K")
    print(f"{'='*70}\n")
    
    # Para la primera temperatura, usar estructura minimizada
    # Para el resto, usar el último frame de la temperatura anterior
    if i == 0:
        atoms = read(minimized_file)
        atoms.calc = calc
        print(f"Estructura inicial: Minimizada")
    else:
        atoms = last_atoms.copy()
        atoms.calc = calc
        print(f"Estructura inicial: Último frame de T={temperatures[i-1]:.1f} K")
    
    
    # Inicializar velocidades con distribución Maxwell-Boltzmann
    MaxwellBoltzmannDistribution(atoms, temperature_K=T)
    
    # ========================================================================
    # FASE 1: EQUILIBRACIÓN NVT (volumen constante) con Langevin
    # ========================================================================
    print(f"⏱️  Equilibrando con Langevin ({equilibration_ps} ps, {equilibration_steps} pasos)...")
    
    equil_traj_file = f"{output_dir}/fau_T{int(T):04d}K_P{pressure_GPa}GPa_equilibration.traj"
    equil_log_file = f"{output_dir}/fau_T{int(T):04d}K_P{pressure_GPa}GPa_equilibration.log"
    
    dyn_equi = Langevin(
        atoms,
        timestep=timestep_fs,
        temperature_K=T,
        friction=0.01 ,
        logfile=equil_log_file,
        trajectory=equil_traj_file,
        loginterval=dump_interval
    )
    
    dyn_equi.run(equilibration_steps)
    
    print(f"  ✓ Equilibración completada")
    print(f"  💾 Trayectoria: {equil_traj_file}\n")
    
    # ========================================================================
    # FASE 2: PRODUCCIÓN MTKNPT (barostato + termostato)
    # ========================================================================
    print(f"📊 Producción con MTKNPT ({production_ps} ps, {production_steps} pasos)...")
    
    # Archivo de trayectoria y log de producción
    traj_file = f"{output_dir}/fau_T{int(T):04d}K_P{pressure_GPa}GPa_production.traj"
    prod_log_file = f"{output_dir}/fau_T{int(T):04d}K_P{pressure_GPa}GPa_production.log"
    
    cell_transpose = atoms.cell.T
    upper_triangular_cell = np.linalg.cholesky(cell_transpose @ cell_transpose.T)
    upper_triangular_cell = upper_triangular_cell.T
    atoms.set_cell(upper_triangular_cell, scale_atoms=True)

    dyn_prod = MTKNPT(
        atoms,
        timestep=timestep_fs,
        temperature_K=T,
        pressure_au=0.0,
        tdamp=tdamp,
        pdamp=pdamp,
        logfile=prod_log_file,
        trajectory=traj_file,
        loginterval=dump_interval
    )
    
    # Acumular estadísticas para calcular medias
    stats = {
        'vol_sum': 0.0,
        'temp_sum': 0.0,
        'epot_sum': 0.0,
        'ekin_sum': 0.0,
        'cell_a_sum': 0.0,
        'cell_b_sum': 0.0,
        'cell_c_sum': 0.0,
        'vol_sq_sum': 0.0,
        'n_samples': 0
    }
    
    def collect_data():
        """Recolecta datos y calcula estadísticas acumulativas"""
        # Obtener propiedades
        vol = atoms.get_volume()
        temp = atoms.get_temperature()
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        cell_params = atoms.cell.cellpar()
        
        # Acumular para medias
        stats['vol_sum'] += vol
        stats['vol_sq_sum'] += vol**2
        stats['temp_sum'] += temp
        stats['epot_sum'] += epot
        stats['ekin_sum'] += ekin
        stats['cell_a_sum'] += cell_params[0]
        stats['cell_b_sum'] += cell_params[1]
        stats['cell_c_sum'] += cell_params[2]
        stats['n_samples'] += 1
    
    dyn_prod.attach(collect_data, interval=dump_interval)
    dyn_prod.run(production_steps)
    
    # Calcular medias
    n = stats['n_samples']
    vol_mean = stats['vol_sum'] / n
    vol_std = np.sqrt((stats['vol_sq_sum'] / n) - vol_mean**2)
    temp_mean = stats['temp_sum'] / n
    epot_mean = stats['epot_sum'] / n
    ekin_mean = stats['ekin_sum'] / n
    cell_a_mean_val = stats['cell_a_sum'] / n
    cell_b_mean_val = stats['cell_b_sum'] / n
    cell_c_mean_val = stats['cell_c_sum'] / n
    
    # Guardar resultados
    volumes_mean.append(vol_mean)
    volumes_std.append(vol_std)
    cell_a_mean.append(cell_a_mean_val)
    cell_b_mean.append(cell_b_mean_val)
    cell_c_mean.append(cell_c_mean_val)
    energies_mean.append(epot_mean)
    energies_std.append(0.0)  # Sin desviación para compatibilidad
    
    # Guardar última estructura para usar en la siguiente temperatura
    last_atoms = atoms.copy()
    
    print(f"\n  ✓ Producción completada")
    print(f"  📊 Resultados (medias):")
    print(f"     Temperatura: {temp_mean:.2f} K (target: {T:.1f} K)")
    print(f"     Volumen: {vol_mean:.4f} ± {vol_std:.4f} Å³")
    print(f"     Energía potencial: {epot_mean:.6f} eV")
    print(f"     Energía cinética: {ekin_mean:.6f} eV")
    print(f"     Parámetros celda: a={cell_a_mean_val:.4f}, b={cell_b_mean_val:.4f}, c={cell_c_mean_val:.4f} Å")
    print(f"  💾 Trayectoria: {traj_file}")
    
    # Guardar última estructura (opcional)
    if SAVE_FINAL_STRUCTURES:
        last_structure = f"{output_dir}/fau_T{int(T):04d}K_last.vasp"
        write(last_structure, atoms)
        print(f"  💾 Estructura final: {last_structure}")

# ============================================================================
# PASO 3: ANÁLISIS Y AJUSTE LINEAL (OPCIONAL)
# ============================================================================

if DO_ANALYSIS:
    print("\n" + "="*70)
    print(" ANÁLISIS DE EXPANSIÓN TÉRMICA")
    print("="*70 + "\n")
    
    # Convertir a arrays numpy
    temperatures = np.array(temperatures)
    volumes_mean = np.array(volumes_mean)
    volumes_std = np.array(volumes_std)
    cell_a_mean = np.array(cell_a_mean)
    cell_b_mean = np.array(cell_b_mean)
    cell_c_mean = np.array(cell_c_mean)
    energies_mean = np.array(energies_mean)
    energies_std = np.array(energies_std)
    
    # Ajuste lineal: V(T) = V₀ + α_V * V₀ * T
    # donde α_V es el coeficiente de expansión volumétrica
    slope_vol, intercept_vol, r_vol, p_vol, stderr_vol = linregress(temperatures, volumes_mean)
    alpha_V = slope_vol / volumes_mean[0]  # Coeficiente volumétrico (1/K)
    alpha_L = alpha_V / 3.0  # Coeficiente lineal aproximado (1/K)
    
    print(f"📊 Ajuste lineal V(T) = V₀ + m·T:")
    print(f"   V₀ (intercepto): {intercept_vol:.4f} Å³")
    print(f"   m (pendiente): {slope_vol:.6f} Å³/K")
    print(f"   R²: {r_vol**2:.6f}")
    print(f"\n📐 Coeficientes de expansión térmica:")
    print(f"   α_V (volumétrico): {alpha_V:.3e} K⁻¹")
    print(f"   α_L (lineal, aprox.): {alpha_L:.3e} K⁻¹")
    
    # Guardar datos
    data_file = f"{output_dir}/fau_expansion_data_P{pressure_GPa}GPa.txt"
    header = (f"Thermal expansion data for FAU\n"
              f"Pressure: {pressure_GPa} GPa\n"
              f"Timestep: {timestep_fs} fs\n"
              f"Equilibration: {equilibration_ps} ps, Production: {production_ps} ps\n"
              f"Alpha_V: {alpha_V:.6e} K^-1\n"
              f"Alpha_L: {alpha_L:.6e} K^-1\n"
              f"T(K) Volume(A^3) Vol_std a(A) b(A) c(A) Energy(eV) E_std")
    data = np.column_stack([temperatures, volumes_mean, volumes_std, 
                            cell_a_mean, cell_b_mean, cell_c_mean,
                            energies_mean, energies_std])
    np.savetxt(data_file, data, header=header, fmt='%.6f')
    print(f"\n💾 Datos guardados: {data_file}")
else:
    print("\n📊 Análisis omitido (usa --analyze para realizar análisis y ajuste lineal)")

# ============================================================================
# PASO 4: VISUALIZACIÓN (OPCIONAL)
# ============================================================================

if SAVE_PLOTS:
    if not DO_ANALYSIS:
        print("\n⚠️  ADVERTENCIA: --plot requiere --analyze. Activando análisis...")
        # Convertir a arrays numpy
        temperatures = np.array(temperatures)
        volumes_mean = np.array(volumes_mean)
        volumes_std = np.array(volumes_std)
        cell_a_mean = np.array(cell_a_mean)
        cell_b_mean = np.array(cell_b_mean)
        cell_c_mean = np.array(cell_c_mean)
        energies_mean = np.array(energies_mean)
        energies_std = np.array(energies_std)
        
        # Ajuste lineal
        slope_vol, intercept_vol, r_vol, p_vol, stderr_vol = linregress(temperatures, volumes_mean)
        alpha_V = slope_vol / volumes_mean[0]
        alpha_L = alpha_V / 3.0
    
    print("\n📊 Generando gráficas...")
    
    # Figura 1: Expansión volumétrica con ajuste
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(temperatures, volumes_mean, yerr=volumes_std, 
                fmt='o', markersize=8, capsize=5, capthick=2,
                label='Datos MD', color='blue', ecolor='lightblue')
    
    # Línea de ajuste
    T_fit = np.linspace(temperatures.min(), temperatures.max(), 100)
    V_fit = intercept_vol + slope_vol * T_fit
    ax.plot(T_fit, V_fit, '--', color='red', linewidth=2,
            label=f'Ajuste lineal (R²={r_vol**2:.4f})')
    
    ax.set_xlabel('Temperatura (K)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Volumen (Å³)', fontsize=13, fontweight='bold')
    ax.set_title(f'Expansión Térmica FAU (P = {pressure_GPa} GPa)\n' + 
                 f'α_V = {alpha_V:.3e} K⁻¹, α_L = {alpha_L:.3e} K⁻¹',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    vol_plot = f"{output_dir}/fau_volume_vs_temp_P{pressure_GPa}GPa.png"
    plt.savefig(vol_plot, dpi=300, bbox_inches='tight')
    print(f"💾 Gráfica de volumen: {vol_plot}")
    plt.close()
    
    # Figura 2: Parámetros de celda
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(temperatures, cell_a_mean, 'o-', label='a', markersize=6, linewidth=2)
    ax.plot(temperatures, cell_b_mean, 's-', label='b', markersize=6, linewidth=2)
    ax.plot(temperatures, cell_c_mean, '^-', label='c', markersize=6, linewidth=2)
    ax.set_xlabel('Temperatura (K)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Parámetro de celda (Å)', fontsize=13, fontweight='bold')
    ax.set_title(f'Evolución de Parámetros de Celda - FAU (P = {pressure_GPa} GPa)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    cell_plot = f"{output_dir}/fau_cell_params_vs_temp_P{pressure_GPa}GPa.png"
    plt.savefig(cell_plot, dpi=300, bbox_inches='tight')
    print(f"💾 Gráfica de parámetros: {cell_plot}")
    plt.close()
    
    # Figura 3: Energía vs Temperatura
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(temperatures, energies_mean, yerr=energies_std,
                fmt='o-', markersize=8, capsize=5, capthick=2,
                color='green', ecolor='lightgreen', linewidth=2)
    ax.set_xlabel('Temperatura (K)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Energía Potencial (eV)', fontsize=13, fontweight='bold')
    ax.set_title(f'Energía Potencial vs Temperatura - FAU (P = {pressure_GPa} GPa)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    energy_plot = f"{output_dir}/fau_energy_vs_temp_P{pressure_GPa}GPa.png"
    plt.savefig(energy_plot, dpi=300, bbox_inches='tight')
    print(f"💾 Gráfica de energía: {energy_plot}")
    plt.close()
elif DO_ANALYSIS:
    print("\n📊 Visualización omitida (usa --plot para generar gráficas)")
else:
    print("\n📊 Visualización y análisis omitidos (usa --analyze y --plot para activar)")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*70)
print(" ✅ SIMULACIÓN COMPLETADA")
print("="*70)
if DO_ANALYSIS:
    print(f"\nCoeficientes de expansión térmica:")
    print(f"  α_V (volumétrico): {alpha_V:.6e} K⁻¹")
    print(f"  α_L (lineal): {alpha_L:.6e} K⁻¹")
print(f"\nArchivos generados en: {output_dir}/")
print(f"  - fau_T****K_P{pressure_GPa}GPa_equilibration.traj/log: Equilibración")
print(f"  - fau_T****K_P{pressure_GPa}GPa_production.traj/log: Producción MTKNPT")
if SAVE_FINAL_STRUCTURES:
    print(f"  - fau_T****K_last.vasp: Últimas estructuras")
if DO_ANALYSIS:
    print(f"  - {os.path.basename(data_file)}: Datos numéricos")
if SAVE_PLOTS:
    print(f"  - fau_volume_vs_temp_P{pressure_GPa}GPa.png: Gráfica volumen vs T")
    print(f"  - fau_cell_params_vs_temp_P{pressure_GPa}GPa.png: Gráfica parámetros celda")
    print(f"  - fau_energy_vs_temp_P{pressure_GPa}GPa.png: Gráfica energía vs T")
print("\n" + "="*70 + "\n")

