#!/usr/bin/env python3
"""
Script para realizar dinámica molecular de AFI y generar histograma de ángulos
Equilibración: NVT
Producción: MTKNPT
"""
import torch
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.nose_hoover_chain import MTKNPT
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from mace.calculators import MACECalculator
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# PARÁMETROS DE SIMULACIÓN
# ============================================================================

# Archivos de entrada
model_path = "../../zeolite-mh-finetuning.model"
input_structure = "../structures/CONTCAR_AFI.vasp"

# Parámetros MD
temperature_K = 300.0          # Temperatura en Kelvin
pressure_eV_A3 = 0.0           # Presión en eV/Å³ (para fase de producción NPT)
timestep_fs = 0.4              # Paso de tiempo en fs
equilibration_ps = 30.0        # Tiempo de equilibración NVT en ps
#equilibration_ps = 1.0        # Tiempo de equilibración NVT en ps (test)
production_ps = 200.0          # Tiempo de producción MTKNPT en ps
#production_ps = 2.0          # Tiempo de producción MTKNPT en ps (test)
dump_interval = 200             # Guardar cada N pasos


# Constantes de tiempo del termostato y barostato
ttime_fs = 200 * timestep_fs           # Constante de tiempo del termostato (fs)
pfactor = 500 * ttime_fs               # Constante de tiempo del barostato (fs²)

# Configuración del dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# Crear directorio de outputs
output_dir = "outputs_md_angles_good_thermostat"
os.makedirs(output_dir, exist_ok=True)


# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

print("="*70)
print(" DINÁMICA MOLECULAR - ANÁLISIS DE ÁNGULOS AFI")
print("="*70)
print(f"\n📁 Directorio de salida: {output_dir}")
print(f"\n📂 Archivos de entrada:")
print(f"  Estructura: {input_structure}")
print(f"  Modelo MACE: {model_path}")
print(f"\n⚙️  Parámetros MD:")
print(f"  Temperatura: {temperature_K} K")
print(f"  Presión (producción): {pressure_eV_A3} eV/Å³")
print(f"  Timestep: {timestep_fs} fs")
print(f"  Equilibración (NVT): {equilibration_ps} ps")
print(f"  Producción (MTKNPT): {production_ps} ps")
print(f"  Tiempo total: {(equilibration_ps + production_ps):.4f} ps")
print(f"  Intervalo de guardado: cada {dump_interval} pasos")
print(f"\n⚙️  Parámetros termostato/barostato:")
print(f"  ttime: {ttime_fs} fs")
print(f"  pfactor: {pfactor} fs²")
print(f"\n💻 Configuración:")
print(f"  CuEq: Activado")
print(f"  Device: {device}")
print("="*70 + "\n")

# Leer estructura
atoms = read(input_structure)

# Inicializar calculador MACE
calc = MACECalculator(
    model_paths=model_path,
    device=device,
    default_dtype="float64",
    enable_cueq=True
)
atoms.calc = calc

# Información inicial
print("📊 Estructura inicial:")
cell_params = atoms.cell.cellpar()
print(f"  Átomos: {len(atoms)}")
print(f"  Celda: a={cell_params[0]:.3f}, b={cell_params[1]:.3f}, c={cell_params[2]:.3f} Å")
print(f"  Ángulos: α={cell_params[3]:.2f}°, β={cell_params[4]:.2f}°, γ={cell_params[5]:.2f}°")
print(f"  Volumen: {atoms.get_volume():.2f} Å³")
print(f"  Energía inicial: {atoms.get_potential_energy():.4f} eV\n")

# ============================================================================
# FASE 1: EQUILIBRACIÓN NVT
# ============================================================================

print("🔄 FASE 1: EQUILIBRACIÓN NVT")
print("-"*70)

# Inicializar velocidades según distribución de Maxwell-Boltzmann
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)


# Archivo de log para equilibración
equi_log_file = f"{output_dir}/afi_md_equilibration_T{int(temperature_K)}K_NVT.log"
equi_traj_file = f"{output_dir}/afi_md_equilibration_T{int(temperature_K)}K_NVT.traj"
# Crear dinámica NVT para equilibración (usando Langevin)
dyn_equi = Langevin(
    atoms,
    timestep=timestep_fs,
    temperature_K=temperature_K,
    friction=0.002,
    logfile=equi_log_file,
    loginterval=dump_interval,
    trajectory=equi_traj_file
)

# Equilibración (sin prints para máximo rendimiento)
equilibration_steps = int(equilibration_ps * 1000 / timestep_fs)        # Pasamos a fs los ps y luego calculamos pasos
print(f"Equilibrando por {equilibration_steps} pasos ({equilibration_ps} ps)...\n")

dyn_equi.run(equilibration_steps)

print(f"\n✓ Equilibración completada\n")

# ============================================================================
# FASE 2: PRODUCCIÓN MTKNPT
# ============================================================================

print("🔄 FASE 2: PRODUCCIÓN MTKNPT")
print("-"*70)

production_steps = int(production_ps * 1000 / timestep_fs)
print(f"Simulación de producción: {production_steps} pasos ({production_ps} ps)...\n")

# Preallocar arrays para mejor rendimiento
n_samples = production_steps // dump_interval + 1
angles_alpha = np.zeros(n_samples)
angles_beta = np.zeros(n_samples)
angles_gamma = np.zeros(n_samples)
volumes = np.zeros(n_samples)
temperatures = np.zeros(n_samples)
energies_pot = np.zeros(n_samples)
energies_kin = np.zeros(n_samples)
times = np.zeros(n_samples)
data_counter = 0

# Archivo de trayectoria
traj_file = f"{output_dir}/afi_md_T{int(temperature_K)}K_P{pressure_eV_A3}eVA3.traj"
traj = Trajectory(traj_file, 'w', atoms)

# Archivo de log para producción
prod_log_file = f"{output_dir}/afi_md_production_T{int(temperature_K)}K_P{pressure_eV_A3}eVA3_MTKNPT.log"


# Crear dinámica MTKNPT para producción
dyn_prod = MTKNPT(
    atoms,
    timestep=timestep_fs,
    temperature_K=temperature_K,
    pressure_au=pressure_eV_A3,
    tdamp=ttime_fs,
    pdamp=pfactor,
    logfile=prod_log_file,
    loginterval=dump_interval
)

def collect_data():
    """Recolecta datos durante la simulación (optimizado)"""
    global data_counter
    
    # Obtener parámetros de celda de una vez
    cell_params = atoms.cell.cellpar()
    angles_alpha[data_counter] = cell_params[3]
    angles_beta[data_counter] = cell_params[4]
    angles_gamma[data_counter] = cell_params[5]
    
    # Propiedades termodinámicas
    volumes[data_counter] = atoms.get_volume()
    temperatures[data_counter] = atoms.get_temperature()
    energies_pot[data_counter] = atoms.get_potential_energy()
    energies_kin[data_counter] = atoms.get_kinetic_energy()
    times[data_counter] = dyn_prod.nsteps * timestep_fs / 1000.0
    
    data_counter += 1
    
    # Guardar trayectoria
    traj.write()

# Ejecutar producción
dyn_prod.attach(collect_data, interval=dump_interval)
dyn_prod.run(production_steps)
traj.close()

print(f"\n✓ Producción completada")
print(f"💾 Trayectoria guardada: {traj_file}")
print(f"💾 Log de producción guardado: {prod_log_file}\n")

# ============================================================================
# ANÁLISIS Y VISUALIZACIÓN
# ============================================================================

print("📊 GENERANDO ANÁLISIS Y GRÁFICAS...")
print("-"*70)

# Recortar arrays al tamaño real usado
times = times[:data_counter]
angles_alpha = angles_alpha[:data_counter]
angles_beta = angles_beta[:data_counter]
angles_gamma = angles_gamma[:data_counter]
volumes = volumes[:data_counter]
temperatures = temperatures[:data_counter]
energies_pot = energies_pot[:data_counter]
energies_kin = energies_kin[:data_counter]

# Estadísticas
print(f"\nEstadísticas de los ángulos:")
print(f"  α: {np.mean(angles_alpha):.4f} ± {np.std(angles_alpha):.4f}°  "
      f"[{np.min(angles_alpha):.4f}, {np.max(angles_alpha):.4f}]")
print(f"  β: {np.mean(angles_beta):.4f} ± {np.std(angles_beta):.4f}°  "
      f"[{np.min(angles_beta):.4f}, {np.max(angles_beta):.4f}]")
print(f"  γ: {np.mean(angles_gamma):.4f} ± {np.std(angles_gamma):.4f}°  "
      f"[{np.min(angles_gamma):.4f}, {np.max(angles_gamma):.4f}]")

print(f"\nEstadísticas termodinámicas:")
print(f"  Temperatura: {np.mean(temperatures):.2f} ± {np.std(temperatures):.2f} K")
print(f"  Volumen: {np.mean(volumes):.2f} ± {np.std(volumes):.2f} Å³")
print(f"  Energía potencial: {np.mean(energies_pot):.4f} ± {np.std(energies_pot):.4f} eV")

# Guardar datos en archivo
data_file = f"{output_dir}/afi_md_data_T{int(temperature_K)}K.txt"
header = "Time(ps) Alpha(deg) Beta(deg) Gamma(deg) Volume(A^3) Temp(K) Epot(eV) Ekin(eV)"
data = np.column_stack([times, angles_alpha, angles_beta, angles_gamma, 
                        volumes, temperatures, energies_pot, energies_kin])
np.savetxt(data_file, data, header=header, fmt='%.6f')
print(f"💾 Datos guardados: {data_file}")

# ============================================================================
# GRÁFICAS (Optimizadas para rendimiento)
# ============================================================================

# Control de generación de gráficas (puedes desactivar para máximo rendimiento)
GENERATE_PLOTS = False

if GENERATE_PLOTS:
    print("Generando gráficas...")
else:
    print("Gráficas desactivadas para optimizar rendimiento")

# Figura 1: Histogramas de ángulos
if GENERATE_PLOTS:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Histogramas de Ángulos - AFI MD NPT (T={temperature_K}K, P={pressure_eV_A3}eV/Å³)', 
                 fontsize=14, fontweight='bold')

    angles_data = [angles_alpha, angles_beta, angles_gamma]
    angle_names = ['α (alpha)', 'β (beta)', 'γ (gamma)']
    colors = ['red', 'blue', 'green']

    for i, (ax, angle, name, color) in enumerate(zip(axes, angles_data, angle_names, colors)):
        ax.hist(angle, bins=50, alpha=0.7, color=color, edgecolor='black')
        ax.axvline(np.mean(angle), color='black', linestyle='--', linewidth=2, 
                   label=f'Media: {np.mean(angle):.3f}°')
        #ax.axvline(90.0, color='gray', linestyle=':', linewidth=2, alpha=0.5, label='90°')
        ax.set_xlabel(f'Ángulo {name} (grados)', fontsize=11)
        ax.set_ylabel('Frecuencia', fontsize=11)
        ax.set_title(f'{name}: {np.mean(angle):.3f} ± {np.std(angle):.3f}°', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    hist_file = f"{output_dir}/afi_angles_histogram_T{int(temperature_K)}K.png"
    plt.savefig(hist_file, dpi=150, bbox_inches='tight')  # DPI reducido para velocidad
    print(f"💾 Histograma guardado: {hist_file}")
    plt.close()

# Figura 2: Evolución temporal de ángulos (3 subfiguras)
if GENERATE_PLOTS:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(f'Evolución Temporal de Ángulos - AFI (T={temperature_K}K, P={pressure_eV_A3}eV/Å³)',
                 fontsize=14, fontweight='bold')

    angle_list = [(angles_alpha, 'α (alpha)', 'red'),
                  (angles_beta, 'β (beta)', 'blue'),
                  (angles_gamma, 'γ (gamma)', 'green')]

    for ax, (angle, name, color) in zip(axes, angle_list):
        ax.plot(times, angle, label=name, color=color, alpha=0.8, linewidth=1)
        ax.axhline(np.mean(angle), color='black', linestyle='--', linewidth=1,
                   label=f'Media: {np.mean(angle):.3f}°')
        #ax.axhline(90.0, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='90° (referencia)')
        ax.set_xlabel('Tiempo (ps)', fontsize=11)
        ax.set_ylabel('Ángulo (grados)', fontsize=11)
        ax.set_title(f'{name}: {np.mean(angle):.3f} ± {np.std(angle):.3f}°', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    evolution_file = f"{output_dir}/afi_angles_evolution_T{int(temperature_K)}K.png"
    plt.savefig(evolution_file, dpi=150, bbox_inches='tight')  # DPI reducido para velocidad
    print(f"💾 Evolución temporal guardada: {evolution_file}")
    plt.close()

# Figura 3: Propiedades termodinámicas
if GENERATE_PLOTS:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Propiedades Termodinámicas - AFI MD NPT (T={temperature_K}K, P={pressure_eV_A3}eV/Å³)', 
                 fontsize=14, fontweight='bold')

    # Temperatura
    axes[0, 0].plot(times, temperatures, color='orange', linewidth=1)
    axes[0, 0].axhline(temperature_K, color='red', linestyle='--', label=f'Target: {temperature_K}K')
    axes[0, 0].set_xlabel('Tiempo (ps)', fontsize=11)
    axes[0, 0].set_ylabel('Temperatura (K)', fontsize=11)
    axes[0, 0].set_title(f'Temperatura: {np.mean(temperatures):.2f} ± {np.std(temperatures):.2f} K')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Volumen
    axes[0, 1].plot(times, volumes, color='purple', linewidth=1)
    axes[0, 1].set_xlabel('Tiempo (ps)', fontsize=11)
    axes[0, 1].set_ylabel('Volumen (Å³)', fontsize=11)
    axes[0, 1].set_title(f'Volumen: {np.mean(volumes):.2f} ± {np.std(volumes):.2f} Å³')
    axes[0, 1].grid(True, alpha=0.3)

    # Energía potencial
    axes[1, 0].plot(times, energies_pot, color='blue', linewidth=1)
    axes[1, 0].set_xlabel('Tiempo (ps)', fontsize=11)
    axes[1, 0].set_ylabel('Energía Potencial (eV)', fontsize=11)
    axes[1, 0].set_title(f'E_pot: {np.mean(energies_pot):.4f} ± {np.std(energies_pot):.4f} eV')
    axes[1, 0].grid(True, alpha=0.3)

    # Energía total
    energies_total = energies_pot + energies_kin
    axes[1, 1].plot(times, energies_total, color='green', linewidth=1)
    axes[1, 1].set_xlabel('Tiempo (ps)', fontsize=11)
    axes[1, 1].set_ylabel('Energía Total (eV)', fontsize=11)
    axes[1, 1].set_title(f'E_total: {np.mean(energies_total):.4f} ± {np.std(energies_total):.4f} eV')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    thermo_file = f"{output_dir}/afi_thermodynamics_T{int(temperature_K)}K.png"
    plt.savefig(thermo_file, dpi=150, bbox_inches='tight')  # DPI reducido para velocidad
    print(f"💾 Propiedades termodinámicas guardadas: {thermo_file}")
    plt.close()

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*70)
print(" ✅ SIMULACIÓN COMPLETADA")
print("="*70)
print(f"\nArchivos generados en: {output_dir}/")
print(f"  - {os.path.basename(traj_file)}: Trayectoria completa")
print(f"  - {os.path.basename(data_file)}: Datos numéricos")
if GENERATE_PLOTS:
    print(f"  - {os.path.basename(hist_file)}: Histogramas de ángulos")
    print(f"  - {os.path.basename(evolution_file)}: Evolución temporal")
    print(f"  - {os.path.basename(thermo_file)}: Propiedades termodinámicas")
print("\n" + "="*70 + "\n")

