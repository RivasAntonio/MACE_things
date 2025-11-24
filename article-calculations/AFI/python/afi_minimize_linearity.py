#!/usr/bin/env python3
"""
Script para minimizar estructuras AFI y verificar la linealidad de los ángulos en el eje c.
Compara CONTCAR_AFI.vasp vs CONTCAR_AFI_MS_linear.vasp
"""

from ase.io import read, write
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter
from mace.calculators import MACECalculator
import numpy as np
import os

# Configuración
model_path = "../../zeolite-mh-finetuning.model"
pressure_gpa = 0.0  # Presión en GPa
fmax = 0.01  # Criterio de convergencia (eV/Å)

# Crear directorio de outputs
output_dir = "outputs_minimization"
os.makedirs(output_dir, exist_ok=True)
print(f"📁 Directorio de salida: {output_dir}\n")

# Inicializar calculador MACE con CuEq activado
calc = MACECalculator(
    model_paths=model_path,
    device="cuda",
    default_dtype="float64",
    enable_cueq=True
)

def analyze_angles(atoms, label):
    """Analiza los ángulos de la celda"""
    cell_params = atoms.cell.cellpar()
    a, b, c, alpha, beta, gamma = cell_params
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Parámetros de celda:")
    print(f"    a = {a:.4f} Å")
    print(f"    b = {b:.4f} Å")
    print(f"    c = {c:.4f} Å")
    print(f"    α = {alpha:.4f}°")
    print(f"    β = {beta:.4f}°")
    print(f"    γ = {gamma:.4f}°")
    print(f"  Volumen: {atoms.get_volume():.4f} Å³")
    print(f"  Energía: {atoms.get_potential_energy():.6f} eV")
    print(f"  Energía por átomo: {atoms.get_potential_energy()/len(atoms):.6f} eV/atom")
    
    # Verificar linealidad (ángulos cercanos a 90°)
    angle_deviation = max(abs(alpha - 90), abs(beta - 90), abs(gamma - 90))
    is_linear = angle_deviation < 1.0  # Tolerancia de 1 grado
    print(f"\n  Desviación máxima de 90°: {angle_deviation:.4f}°")
    print(f"  ¿Celda ortogonal?: {'✓ SÍ' if is_linear else '✗ NO'}")
    print(f"{'='*60}\n")
    
    return cell_params

def minimize_structure(input_file, output_prefix):
    """Minimiza una estructura a presión constante"""
    
    print(f"\n{'#'*70}")
    print(f"# Procesando: {input_file}")
    print(f"{'#'*70}\n")
    
    # Leer estructura
    atoms = read(input_file)
    atoms.calc = calc
    
    # Analizar estado inicial
    print("📊 ESTADO INICIAL:")
    initial_params = analyze_angles(atoms, "Antes de minimización")
    
    # Configurar minimización con presión externa
    pressure_ev_ang3 = pressure_gpa * 1.602176634  # Convertir GPa a eV/Å³
    ucf = UnitCellFilter(atoms, scalar_pressure=pressure_ev_ang3)
    
    # Archivos de salida
    traj_file = f"{output_dir}/{output_prefix}_minimization.traj"
    log_file = f"{output_dir}/{output_prefix}_minimization.log"
    
    # Optimizar
    print(f"🔄 MINIMIZANDO (P = {pressure_gpa} GPa, fmax = {fmax} eV/Å)...")
    print(f"   Trayectoria: {traj_file}")
    print(f"   Log: {log_file}")
    
    opt = BFGS(ucf, trajectory=traj_file, logfile=log_file)
    opt.run(fmax=fmax)
    
    # Analizar estado final
    print("\n📊 ESTADO FINAL:")
    final_params = analyze_angles(atoms, "Después de minimización")
    
    # Guardar estructura optimizada
    output_files = {
        'vasp': f"{output_dir}/{output_prefix}_minimized.vasp",
        'xyz': f"{output_dir}/{output_prefix}_minimized.xyz",
        'cif': f"{output_dir}/{output_prefix}_minimized.cif"
    }
    
    for fmt, filepath in output_files.items():
        write(filepath, atoms, format=fmt)
        print(f"💾 Guardado: {filepath}")
    
    # Resumen de cambios
    print(f"\n{'='*60}")
    print(f"  CAMBIOS EN PARÁMETROS DE CELDA")
    print(f"{'='*60}")
    labels = ['a (Å)', 'b (Å)', 'c (Å)', 'α (°)', 'β (°)', 'γ (°)']
    for i, label in enumerate(labels):
        change = final_params[i] - initial_params[i]
        print(f"  {label:8s}: {initial_params[i]:8.4f} → {final_params[i]:8.4f}  (Δ = {change:+8.4f})")
    print(f"{'='*60}\n")
    
    return atoms, initial_params, final_params


# ============================================================================
# MAIN: Minimizar ambas estructuras
# ============================================================================

print("\n" + "="*70)
print(" MINIMIZACIÓN DE ESTRUCTURAS AFI - ANÁLISIS DE LINEALIDAD")
print("="*70)
print(f"\nModelo: {model_path}")
print(f"Presión: {pressure_gpa} GPa")
print(f"Criterio convergencia: {fmax} eV/Å")
print(f"CuEq: Activado")
print(f"Device: CUDA")
print("\n" + "="*70 + "\n")

# Estructura 1: AFI regular
atoms_afi, initial_afi, final_afi = minimize_structure(
    "../structures/CONTCAR_AFI.vasp",
    "AFI"
)

# Estructura 2: AFI con ángulos lineales forzados en MS
atoms_afi_linear, initial_afi_linear, final_afi_linear = minimize_structure(
    "../structures/CONTCAR_AFI_MS_linear.vasp",
    "AFI_MS_linear"
)

# ============================================================================
# COMPARACIÓN FINAL
# ============================================================================

print("\n" + "="*70)
print(" COMPARACIÓN FINAL DE ESTRUCTURAS")
print("="*70)

energy_afi = atoms_afi.get_potential_energy()
energy_afi_linear = atoms_afi_linear.get_potential_energy()
energy_diff = energy_afi_linear - energy_afi

print(f"\nEnergías finales:")
print(f"  AFI:          {energy_afi:.6f} eV  ({energy_afi/len(atoms_afi):.6f} eV/atom)")
print(f"  AFI_MS_linear: {energy_afi_linear:.6f} eV  ({energy_afi_linear/len(atoms_afi_linear):.6f} eV/atom)")
print(f"\nDiferencia energética:")
print(f"  ΔE = {energy_diff:.6f} eV  ({energy_diff/len(atoms_afi):.6f} eV/atom)")

# Verificar si ambas convergen a ángulos lineales
def is_orthogonal(params, tol=1.0):
    """Verifica si los ángulos son ortogonales (lineales)"""
    alpha, beta, gamma = params[3], params[4], params[5]
    return all(abs(angle - 90) < tol for angle in [alpha, beta, gamma])

afi_orthogonal = is_orthogonal(final_afi)
afi_linear_orthogonal = is_orthogonal(final_afi_linear)

print(f"\n¿Celdas ortogonales después de minimización?")
print(f"  AFI:           {'✓ SÍ' if afi_orthogonal else '✗ NO'}")
print(f"  AFI_MS_linear: {'✓ SÍ' if afi_linear_orthogonal else '✗ NO'}")

print("\n" + "="*70)
print(" ✅ ANÁLISIS COMPLETADO")
print("="*70)
print(f"\nTodos los archivos guardados en: {output_dir}/")
print("\nArchivos generados:")
print("  - *.vasp, *.xyz, *.cif: Estructuras optimizadas")
print("  - *.traj: Trayectorias de optimización")
print("  - *.log: Logs de optimización")
print("\n" + "="*70 + "\n")

