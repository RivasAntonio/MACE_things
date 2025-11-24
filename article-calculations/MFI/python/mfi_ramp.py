#!/usr/bin/env python3
"""
Script para realizar un rampeo térmico lineal de 280 K a 420 K en la estructura MFI monoclinic
utilizando dinámica molecular NPT con celda variable (anisotropic).

Equivalente ASE del script LAMMPS: mfi_thermal_ramp.in
"""

from ase.io import read, write
from ase import units
from mace.calculators import MACECalculator
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nose_hoover_chain import MTKNPT
import numpy as np
import os

# ============================================================================
# PARÁMETROS
# ============================================================================

# Ruta al modelo
model_path = "../../zeolite-mh-finetuning.model"

# Estructura inicial
structure_file = "../structures/CONTCAR_MFI_monoclinic.vasp"

# Parámetros de temperatura
T_start = 280.0  # K - Temperatura inicial
T_end = 420.0    # K - Temperatura final

# Parámetros de simulación
timestep = 0.4  # fs (0.0004 ps en units metal de LAMMPS)
n_steps = 500000  # Número total de pasos MD
thermo_freq = 100  # Frecuencia de escritura (cada 100 pasos)

# Parámetros NPT
pressure_gpa = 0.0  # Presión externa en GPa
tdamp = 50.0 * timestep * units.fs  # Damping del termostato (10*dt)
pdamp = 500.0 * timestep * units.fs  # Damping del barostato (100*dt)

# Directorio de salida
output_dir = "outputs_thermal_ramp_good_thermobarostat"
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

print("="*70)
print(" RAMPEO TÉRMICO MFI MONOCLINIC: NPT ANISOTROPIC")
print("="*70)
print(f"\nModelo: {model_path}")
print(f"Estructura: {structure_file}")
print(f"Temperatura inicial: {T_start} K")
print(f"Temperatura final: {T_end} K")
print(f"Presión: {pressure_gpa} GPa")
print(f"Timestep: {timestep} fs")
print(f"Pasos totales: {n_steps}")
print(f"Tiempo total: {n_steps * timestep / 1000:.2f} ps")
print(f"Damping termostato: {tdamp / units.fs:.2f} fs")
print(f"Damping barostato: {pdamp / units.fs:.2f} fs")
print(f"Frecuencia de guardado: cada {thermo_freq} pasos")
print(f"Device: CUDA")
print(f"CuEq: Activado")
print(f"Directorio de salida: {output_dir}")
print("="*70 + "\n")

# ============================================================================
# LECTURA DE ESTRUCTURA Y CONFIGURACIÓN DEL CALCULADOR
# ============================================================================

print("📖 Leyendo estructura...")
atoms = read(structure_file)

# Información inicial
cell_params_initial = atoms.cell.cellpar()
volume_initial = atoms.get_volume()
n_atoms = len(atoms)

print(f"\n📊 ESTADO INICIAL:")
print("-"*70)
print(f"  Número de átomos: {n_atoms}")
print(f"  Composición: {atoms.get_chemical_formula()}")
print(f"  Parámetros de celda:")
print(f"    a = {cell_params_initial[0]:.6f} Å")
print(f"    b = {cell_params_initial[1]:.6f} Å")
print(f"    c = {cell_params_initial[2]:.6f} Å")
print(f"    α = {cell_params_initial[3]:.4f}°")
print(f"    β = {cell_params_initial[4]:.4f}°")
print(f"    γ = {cell_params_initial[5]:.4f}°")
print(f"  Volumen: {volume_initial:.6f} Å³")

# Configurar calculador MACE
print(f"\n⚙️  Configurando calculador MACE...")
calc = MACECalculator(
    model_paths=model_path,
    device="cuda",
    default_dtype="float64",
    enable_cueq=True
)
atoms.calc = calc

# Calcular energía inicial
energy_initial = atoms.get_potential_energy()
print(f"  Energía inicial: {energy_initial:.6f} eV")
print(f"  Energía por átomo: {energy_initial/n_atoms:.6f} eV/atom")

# ============================================================================
# CONFIGURACIÓN DE ARCHIVOS DE SALIDA
# ============================================================================

# Archivo para datos de beta vs temperatura
beta_file = f"{output_dir}/mfi_beta_vs_ramp.txt"

print(f"\n📁 Archivos de salida:")
print(f"  Trayectoria ASE: {output_dir}/mfi_thermal_ramp.traj")
print(f"  Log MD:          {output_dir}/mfi_thermal_ramp.log")
print(f"  Beta vs T:       {beta_file}")

# Inicializar archivo de beta
with open(beta_file, 'w') as f:
    f.write("# step time(ps) temp_target(K) temp_actual(K) beta(deg) a(Ang) b(Ang) c(Ang) "
            "alpha(deg) gamma(deg) volume(Ang^3) energy(eV)\n")

# ============================================================================
# FUNCIÓN DE CALLBACK PARA RAMPA DE TEMPERATURA Y REGISTRO
# ============================================================================

step_counter = [0]

def write_beta_data():
    """
    Función que se llama cada thermo_freq pasos para:
    1. Actualizar la temperatura objetivo (rampa lineal)
    2. Escribir datos de beta y parámetros de celda
    """
    step = step_counter[0]
    
    # Calcular temperatura objetivo para este paso (rampa lineal)
    fraction = step / n_steps
    T_target = T_start + (T_end - T_start) * fraction
    
    # Actualizar la temperatura del termostato (MTKNPT usa _temperature_K directamente)
    dyn._temperature_K = T_target
    
    # Obtener propiedades actuales
    T_actual = atoms.get_temperature()
    cell_params = atoms.cell.cellpar()
    a, b, c = cell_params[0:3]
    alpha, beta, gamma = cell_params[3:6]
    volume = atoms.get_volume()
    energy = atoms.get_potential_energy()
    time_ps = step * timestep / 1000.0
    
    # Escribir a archivo de beta
    with open(beta_file, 'a') as f:
        f.write(f"{step} {time_ps:.4f} {T_target:.2f} {T_actual:.2f} "
                f"{beta:.6f} {a:.6f} {b:.6f} {c:.6f} "
                f"{alpha:.4f} {gamma:.4f} {volume:.6f} {energy:.6f}\n")
    
    # Imprimir progreso en pantalla cada 1000 pasos
    if step % (thermo_freq * 10) == 0:
        progress = 100.0 * step / n_steps
        print(f"Paso {step:7d}/{n_steps} ({progress:5.1f}%) | "
              f"T = {T_target:6.1f} K | β = {beta:7.4f}° | "
              f"V = {volume:10.2f} Å³ | E = {energy:12.4f} eV")
    
    step_counter[0] += 1

# ============================================================================
# CONFIGURACIÓN DE LA DINÁMICA MOLECULAR NPT
# ============================================================================

print(f"\n🔄 Configurando MTKNPT con celda anisotropic...")

# Convertir presión de GPa a eV/Å³
pressure_ev_ang3 = pressure_gpa * 1.602176634

# Crear objeto NPT con Martyna-Tobias-Klein
dyn = MTKNPT(
    atoms,
    timestep=timestep * units.fs,
    temperature_K=T_start,  # Se actualizará en cada paso
    pressure_au=0.0,
    tdamp=tdamp,
    pdamp=pdamp,
    logfile=f"{output_dir}/mfi_thermal_ramp.log",
    trajectory=f"{output_dir}/mfi_thermal_ramp.traj",
    loginterval=thermo_freq
)

# ============================================================================
# EJECUTAR DINÁMICA MOLECULAR
# ============================================================================

print(f"\n{'='*70}")
print(" INICIANDO DINÁMICA MOLECULAR NPT")
print(f"{'='*70}\n")

print(f"{'Paso':>7} {'Progreso':>8} {'T_target(K)':>12} {'Beta(°)':>10} "
      f"{'Volume(Å³)':>12} {'Energía(eV)':>14}")
print("-"*70)

# Adjuntar callback para actualizar temperatura y escribir beta
dyn.attach(write_beta_data, interval=thermo_freq)

# Ejecutar MD
try:
    dyn.run(n_steps)
    print(f"\n✅ Simulación completada exitosamente!")
    
except KeyboardInterrupt:
    print(f"\n⚠️  Simulación interrumpida por el usuario en el paso {step_counter[0]}")
    
except Exception as e:
    print(f"\n❌ Error durante la simulación: {e}")
    raise

# ============================================================================
# GUARDAR ESTRUCTURA FINAL
# ============================================================================

final_structure_file = f"{output_dir}/mfi_thermal_ramp_final.vasp"
write(final_structure_file, atoms)
print(f"\n💾 Estructura final guardada: {final_structure_file}")

# ============================================================================
# ANÁLISIS FINAL
# ============================================================================

cell_params_final = atoms.cell.cellpar()
volume_final = atoms.get_volume()
energy_final = atoms.get_potential_energy()

print(f"\n{'='*70}")
print(" RESUMEN FINAL")
print(f"{'='*70}\n")

print("📊 ESTADO INICIAL:")
print("-"*70)
print(f"  Temperatura: {T_start} K")
print(f"  Parámetros de celda:")
print(f"    a = {cell_params_initial[0]:.6f} Å")
print(f"    b = {cell_params_initial[1]:.6f} Å")
print(f"    c = {cell_params_initial[2]:.6f} Å")
print(f"    α = {cell_params_initial[3]:.4f}°")
print(f"    β = {cell_params_initial[4]:.4f}°")
print(f"    γ = {cell_params_initial[5]:.4f}°")
print(f"  Volumen: {volume_initial:.6f} Å³")
print(f"  Energía: {energy_initial:.6f} eV")

print(f"\n📊 ESTADO FINAL:")
print("-"*70)
print(f"  Temperatura: {T_end} K")
print(f"  Parámetros de celda:")
print(f"    a = {cell_params_final[0]:.6f} Å")
print(f"    b = {cell_params_final[1]:.6f} Å")
print(f"    c = {cell_params_final[2]:.6f} Å")
print(f"    α = {cell_params_final[3]:.4f}°")
print(f"    β = {cell_params_final[4]:.4f}°")
print(f"    γ = {cell_params_final[5]:.4f}°")
print(f"  Volumen: {volume_final:.6f} Å³")
print(f"  Energía: {energy_final:.6f} eV")

print(f"\n📊 CAMBIOS TOTALES:")
print("-"*70)
print(f"  ΔT = {T_end - T_start:+.1f} K")

labels = ['a', 'b', 'c', 'α', 'β', 'γ']
units_label = ['Å', 'Å', 'Å', '°', '°', '°']
for i, (label, unit) in enumerate(zip(labels, units_label)):
    change = cell_params_final[i] - cell_params_initial[i]
    if i < 3:
        change_pct = change / cell_params_initial[i] * 100
        print(f"  Δ{label} = {change:+.6f} {unit} ({change_pct:+.4f} %)")
    else:
        print(f"  Δ{label} = {change:+.6f} {unit}")

vol_change = volume_final - volume_initial
vol_change_pct = vol_change / volume_initial * 100
print(f"  ΔV = {vol_change:+.6f} Å³ ({vol_change_pct:+.4f} %)")

energy_change = energy_final - energy_initial
print(f"  ΔE = {energy_change:+.6f} eV ({energy_change/n_atoms:+.6f} eV/atom)")

# Coeficiente de expansión térmica estimado
# α_V = (1/V) * (dV/dT)
if T_end != T_start:
    alpha_V = (vol_change / volume_initial) / (T_end - T_start)
    print(f"\n📊 COEFICIENTE DE EXPANSIÓN TÉRMICA (estimado):")
    print("-"*70)
    print(f"  α_V = {alpha_V:.6e} K⁻¹")
    print(f"  α_V = {alpha_V * 1e6:.4f} × 10⁻⁶ K⁻¹")

# Cambio en el ángulo beta (importante para la transición monoclinic-orthorhombic)
beta_change = cell_params_final[4] - cell_params_initial[4]
print(f"\n📊 CAMBIO EN ÁNGULO BETA (clave para transición de fase):")
print("-"*70)
print(f"  β inicial: {cell_params_initial[4]:.4f}°")
print(f"  β final:   {cell_params_final[4]:.4f}°")
print(f"  Δβ:        {beta_change:+.4f}°")

# Verificar si hay transición de fase
if abs(cell_params_final[4] - 90.0) < 0.5:
    print(f"  ⚠️  ADVERTENCIA: β ≈ 90° → Posible transición a fase orthorhombic")
else:
    print(f"  ✓ La fase monoclinic se mantiene (β ≠ 90°)")

print(f"\n{'='*70}")
print(" ARCHIVOS GENERADOS")
print(f"{'='*70}\n")
print(f"  📊 Datos beta vs T:       {beta_file}")
print(f"  📝 Log MD:                {output_dir}/mfi_thermal_ramp.log")
print(f"  🎬 Trayectoria ASE:       {output_dir}/mfi_thermal_ramp.traj")
print(f"  🔷 Estructura final:      {final_structure_file}")

print(f"\n{'='*70}")
print(" 🎉 ANÁLISIS COMPLETADO")
print(f"{'='*70}\n")

print("💡 Sugerencias para análisis posterior:")
print("  - Graficar β vs T para ver transición de fase")
print("  - Analizar expansión térmica de cada eje")
print("  - Comparar con datos experimentales")
print("  - Visualizar trayectoria con ASE GUI o OVITO")
print("")

