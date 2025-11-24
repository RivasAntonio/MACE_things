#!/usr/bin/env python3
"""
Script para realizar dinámica molecular NPT a diferentes temperaturas (280-420 K en pasos de 20 K).
Cada temperatura se simula durante 100 ps con celda variable (anisotropic).
"""

from ase.io import read, write
from ase import units
from mace.calculators import MACECalculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.nose_hoover_chain import MTKNPT
import numpy as np
import os
import time


# ============================================================================
# PARÁMETROS
# ============================================================================
start_time = time.time()

# Ruta al modelo
model_path = "../../zeolite-mh-finetuning.model"

# Estructura inicial
structure_file = "../structures/CONTCAR_MFI_monoclinic.vasp"

# Parámetros de temperatura
T_start = 300.0  # K - Temperatura inicial
T_end = 460.0    # K - Temperatura final
T_step = 25.0    # K - Paso de temperatura
temps = np.arange(T_start, T_end + T_step, T_step)

# Parámetros de simulación
timestep = 0.4  # fs (0.0004 ps en units metal de LAMMPS)
time_equilibration = 50.0  # ps - Tiempo de equilibración inicial
time_production = 150.0   # ps - Tiempo de producción
n_steps_equil = int(time_equilibration * 1000 / timestep)
n_steps_prod = int(time_production * 1000 / timestep)
log_interval = 100  # Frecuencia de escritura

# Parámetros NPT
pressure_gpa = 0.0  # Presión externa en GPa
taut = tdamp = 25.0  # Damping del termostato 
taup = pdamp = 50.0  # Damping del barostato 

# Directorio de salida
output_dir = "outputs_thermal_steps_good_tpdamp"
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

print("="*70)
print(" DINÁMICA NPT MFI MONOCLINIC: PASOS DE TEMPERATURA")
print("="*70)
print(f"\nModelo: {model_path}")
print(f"Estructura: {structure_file}")
print(f"Rango de temperatura: {T_start} K - {T_end} K")
print(f"Paso de temperatura: {T_step} K")
print(f"Número de temperaturas: {len(temps)}")
print(f"\n⏱️  Tiempos de simulación:")
print(f"   • Equilibrado: {time_equilibration} ps ({n_steps_equil} pasos)")
print(f"   • Producción: {time_production} ps ({n_steps_prod} pasos)")
print(f"   • Total por temperatura: {time_equilibration + time_production} ps")
print(f"   • Tiempo total: {(time_equilibration + time_production) * len(temps):.1f} ps")
print(f"Presión: {pressure_gpa} GPa")
print(f"Damping termostato: {taut / units.fs:.2f} fs")
print(f"Damping barostato: {taup / units.fs:.2f} fs")
print(f"Frecuencia de guardado: cada {log_interval} pasos")
print(f"Device: CUDA")
print(f"CuEq: Activado")
print(f"Directorio de salida: {output_dir}")
print("="*70 + "\n")

# ============================================================================
# LECTURA DE ESTRUCTURA Y CONFIGURACIÓN DEL CALCULADOR
# ============================================================================

print("📖 Leyendo estructura inicial...")
atoms = read(structure_file)

# Configurar calculador MACE
print(f"\n⚙️  Configurando calculador MACE...")
calc = MACECalculator(
    model_paths=model_path,
    device="cuda",
    default_dtype="float64",
    enable_cueq=True
)
atoms.calc = calc

n_atoms = len(atoms)

# ============================================================================
# BUCLE SOBRE TEMPERATURAS
# ============================================================================

print(f"\n{'='*70}")
print(" INICIANDO DINÁMICAS MOLECULARES NPT")
print(f"{'='*70}\n")


for i, T in enumerate(temps):
    print(f"\n{'─'*70}")
    print(f"  TEMPERATURA {i+1}/{len(temps)}: {T:.0f} K")
    print(f"{'─'*70}")
    
    # ========================================================================
    # FASE DE EQUILIBRADO
    # ========================================================================
    print(f"\n🔧 Fase de equilibrado ({time_equilibration} ps)...")
    
    # Inicializar velocidades según Maxwell-Boltzmann para esta temperatura
    MaxwellBoltzmannDistribution(atoms, temperature_K=T)
    
    # Dinámica de equilibrado
    dyn_equil = MTKNPT(
        atoms,  
        timestep=timestep * units.fs,
        temperature_K=T,
        pressure_au=pressure_gpa * units.GPa,
        tdamp=tdamp * units.fs,
        pdamp=pdamp * units.fs,
        logfile=f"{output_dir}/mfi_{T:.0f}K_equilibration.log",
        trajectory=None,  # No guardar trayectoria del equilibrado
        loginterval=log_interval
    )
    dyn_equil.run(n_steps_equil)
    
    print(f"✅ Equilibrado completado")
    
    # ========================================================================
    # FASE DE PRODUCCIÓN
    # ========================================================================
    print(f"\n🎯 Fase de producción ({time_production} ps)...")
    
    # Dinámica de producción (continúa desde el estado equilibrado)
    dyn_prod = MTKNPT(
        atoms,  
        timestep=timestep * units.fs,
        temperature_K=T,
        pressure_au=pressure_gpa * units.GPa,
        tdamp=tdamp * units.fs,
        pdamp=pdamp * units.fs,
        logfile=f"{output_dir}/mfi_{T:.0f}K_production.log",
        trajectory=f"{output_dir}/mfi_{T:.0f}K_production.traj",
        loginterval=log_interval
    )
    dyn_prod.run(n_steps_prod)
    
    print(f"✅ Producción completada")
    
    # Guardar estructura final de esta temperatura
    write(f"{output_dir}/mfi_{T:.0f}K_final.vasp", atoms)

print(f"\n{'='*70}")
print(" TODAS LAS SIMULACIONES COMPLETADAS")
print(f"{'='*70}\n")

# ============================================================================
# GUARDAR ESTRUCTURA FINAL
# ============================================================================

final_structure_file = f"{output_dir}/mfi_thermal_steps_final.vasp"
write(final_structure_file, atoms)
print(f"💾 Estructura final guardada: {final_structure_file}")

# ============================================================================
# ANÁLISIS FINAL
# ============================================================================
print(f"\n{'='*70}")
print(" ANÁLISIS DE PERFORMANCE")
print(f"{'='*70}")

total_time = time.time() - start_time
total_steps = (n_steps_equil + n_steps_prod) * len(temps)

print(f"\n⏱️  Tiempo total de simulación: {total_time:.2f} s ({total_time/60:.2f} min)")
print(f"\n📊 Estadísticas de performance:")
print(f"   • Número total de átomos: {n_atoms}")
print(f"   • Número total de pasos: {total_steps}")
print(f"   • Tiempo por paso: {total_time/total_steps*1000:.3f} ms")
print(f"   • Tiempo por átomo y paso: {total_time/total_steps/n_atoms*1e6:.3f} µs")
print(f"   • Pasos por segundo: {total_steps/total_time:.2f}")
print(f"   • ns/día (simulación): {timestep * total_steps / total_time * 86400 / 1e6:.2f}")

print(f"\n{'='*70}\n")

