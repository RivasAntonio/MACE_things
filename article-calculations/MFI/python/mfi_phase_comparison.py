#!/usr/bin/env python3
"""
Script para comparar la diferencia energética entre las fases orthorhombic y monoclinic de MFI
mediante minimización a presión 0, permitiendo relajación de la celda.
"""

from ase.io import read, write
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter
from mace.calculators import MACECalculator
import numpy as np
import os

# ============================================================================
# PARÁMETROS
# ============================================================================

model_path = "../../zeolite-mh-finetuning.model"
pressure_gpa = 0.0  # Presión en GPa
fmax = 0.01  # Criterio de convergencia (eV/Å)

# Estructuras a comparar
structures = {
    'orthorhombic': "../structures/CONTCAR_MFI_orthorombic.vasp",
    'monoclinic': "../structures/CONTCAR_MFI_monoclinic.vasp"
}

# Crear directorio de outputs
output_dir = "outputs_phase_comparison"
os.makedirs(output_dir, exist_ok=True)
print(f"📁 Directorio de salida: {output_dir}\n")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

print("="*70)
print(" COMPARACIÓN ENERGÉTICA MFI: ORTHORHOMBIC vs MONOCLINIC")
print("="*70)
print(f"\nModelo: {model_path}")
print(f"Presión: {pressure_gpa} GPa")
print(f"Criterio de convergencia: {fmax} eV/Å")
print(f"CuEq: Activado")
print(f"Device: CUDA")
print("="*70 + "\n")

# Inicializar calculador MACE con CuEq
calc = MACECalculator(
    model_paths=model_path,
    device="cuda",
    default_dtype="float64",
    enable_cueq=True
)

# ============================================================================
# FUNCIÓN DE MINIMIZACIÓN
# ============================================================================

def minimize_and_analyze(structure_file, phase_name):
    """
    Minimiza una estructura y analiza sus propiedades
    """
    
    print(f"\n{'#'*70}")
    print(f"# FASE: {phase_name.upper()}")
    print(f"# Archivo: {structure_file}")
    print(f"{'#'*70}\n")
    
    # Leer estructura
    atoms = read(structure_file)
    atoms.calc = calc
    
    # Información inicial
    print("📊 ESTADO INICIAL:")
    print("-"*70)
    
    cell_params_initial = atoms.cell.cellpar()
    energy_initial = atoms.get_potential_energy()
    volume_initial = atoms.get_volume()
    n_atoms = len(atoms)
    
    print(f"  Número de átomos: {n_atoms}")
    print(f"  Parámetros de celda:")
    print(f"    a = {cell_params_initial[0]:.6f} Å")
    print(f"    b = {cell_params_initial[1]:.6f} Å")
    print(f"    c = {cell_params_initial[2]:.6f} Å")
    print(f"    α = {cell_params_initial[3]:.4f}°")
    print(f"    β = {cell_params_initial[4]:.4f}°")
    print(f"    γ = {cell_params_initial[5]:.4f}°")
    print(f"  Volumen: {volume_initial:.6f} Ų")
    print(f"  Energía total: {energy_initial:.6f} eV")
    print(f"  Energía por átomo: {energy_initial/n_atoms:.6f} eV/atom")
    
    # Verificar tipo de celda
    alpha, beta, gamma = cell_params_initial[3:6]
    is_orthorhombic = all(abs(angle - 90.0) < 0.5 for angle in [alpha, beta, gamma])
    print(f"  Tipo de celda: {'Orthorhombic' if is_orthorhombic else 'Monoclinic'}")
    
    # Configurar minimización
    pressure_ev_ang3 = pressure_gpa * 1.602176634  # Convertir GPa a eV/Å³
    ucf = UnitCellFilter(atoms, scalar_pressure=pressure_ev_ang3)
    
    # Archivos de salida
    traj_file = f"{output_dir}/mfi_{phase_name}_minimization.traj"
    log_file = f"{output_dir}/mfi_{phase_name}_minimization.log"
    
    print(f"\n🔄 MINIMIZACIÓN:")
    print("-"*70)
    print(f"  Presión externa: {pressure_gpa} GPa")
    print(f"  Criterio fmax: {fmax} eV/Å")
    print(f"  Trayectoria: {traj_file}")
    print(f"  Log: {log_file}")
    
    # Optimizar
    opt = BFGS(ucf, trajectory=traj_file, logfile=log_file)
    opt.run(fmax=fmax)
    
    # Información final
    print(f"\n📊 ESTADO FINAL:")
    print("-"*70)
    
    cell_params_final = atoms.cell.cellpar()
    energy_final = atoms.get_potential_energy()
    volume_final = atoms.get_volume()
    
    print(f"  Parámetros de celda:")
    print(f"    a = {cell_params_final[0]:.6f} Å")
    print(f"    b = {cell_params_final[1]:.6f} Å")
    print(f"    c = {cell_params_final[2]:.6f} Å")
    print(f"    α = {cell_params_final[3]:.4f}°")
    print(f"    β = {cell_params_final[4]:.4f}°")
    print(f"    γ = {cell_params_final[5]:.4f}°")
    print(f"  Volumen: {volume_final:.6f} Ų")
    print(f"  Energía total: {energy_final:.6f} eV")
    print(f"  Energía por átomo: {energy_final/n_atoms:.6f} eV/atom")
    
    # Verificar si la simetría se mantiene
    alpha_f, beta_f, gamma_f = cell_params_final[3:6]
    is_orthorhombic_final = all(abs(angle - 90.0) < 0.5 for angle in [alpha_f, beta_f, gamma_f])
    print(f"  Tipo de celda final: {'Orthorhombic' if is_orthorhombic_final else 'Monoclinic'}")
    
    # Cambios
    print(f"\n📊 CAMBIOS:")
    print("-"*70)
    print(f"  ΔE = {energy_final - energy_initial:+.6f} eV")
    print(f"  ΔE/atom = {(energy_final - energy_initial)/n_atoms:+.6f} eV/atom")
    print(f"  ΔV = {volume_final - volume_initial:+.6f} Ų")
    print(f"  ΔV/V₀ = {(volume_final - volume_initial)/volume_initial * 100:+.4f} %")
    
    labels = ['a', 'b', 'c', 'α', 'β', 'γ']
    units_label = ['Å', 'Å', 'Å', '°', '°', '°']
    for i, (label, unit) in enumerate(zip(labels, units_label)):
        change = cell_params_final[i] - cell_params_initial[i]
        change_pct = change / cell_params_initial[i] * 100 if i < 3 else change
        if i < 3:
            print(f"  Δ{label} = {change:+.6f} {unit} ({change_pct:+.4f} %)")
        else:
            print(f"  Δ{label} = {change:+.6f} {unit}")
    
    # Guardar estructuras optimizadas
    output_files = {
        'vasp': f"{output_dir}/mfi_{phase_name}_minimized.vasp",
        'xyz': f"{output_dir}/mfi_{phase_name}_minimized.xyz",
        'cif': f"{output_dir}/mfi_{phase_name}_minimized.cif"
    }
    
    print(f"\n💾 ARCHIVOS GUARDADOS:")
    print("-"*70)
    for fmt, filepath in output_files.items():
        write(filepath, atoms, format=fmt)
        print(f"  {filepath}")
    
    print(f"\n{'#'*70}\n")
    
    # Retornar información relevante
    results = {
        'atoms': atoms,
        'n_atoms': n_atoms,
        'energy_initial': energy_initial,
        'energy_final': energy_final,
        'volume_initial': volume_initial,
        'volume_final': volume_final,
        'cell_initial': cell_params_initial,
        'cell_final': cell_params_final,
        'is_orthorhombic_initial': is_orthorhombic,
        'is_orthorhombic_final': is_orthorhombic_final
    }
    
    return results

# ============================================================================
# MINIMIZAR AMBAS ESTRUCTURAS
# ============================================================================

results = {}

for phase_name, structure_file in structures.items():
    results[phase_name] = minimize_and_analyze(structure_file, phase_name)

# ============================================================================
# COMPARACIÓN FINAL
# ============================================================================

print("\n" + "="*70)
print(" COMPARACIÓN FINAL DE FASES")
print("="*70 + "\n")

ortho = results['orthorhombic']
mono = results['monoclinic']

# Verificar que tienen el mismo número de átomos
assert ortho['n_atoms'] == mono['n_atoms'], "¡Las estructuras tienen diferente número de átomos!"

n_atoms = ortho['n_atoms']

print("📊 ENERGÍAS FINALES:")
print("-"*70)
print(f"  Orthorhombic: {ortho['energy_final']:.6f} eV  ({ortho['energy_final']/n_atoms:.6f} eV/atom)")
print(f"  Monoclinic:   {mono['energy_final']:.6f} eV  ({mono['energy_final']/n_atoms:.6f} eV/atom)")

energy_diff = ortho['energy_final'] - mono['energy_final']
energy_diff_per_atom = energy_diff / n_atoms

print(f"\n📊 DIFERENCIA ENERGÉTICA (Ortho - Mono):")
print("-"*70)
print(f"  ΔE = {energy_diff:+.6f} eV")
print(f"  ΔE/atom = {energy_diff_per_atom:+.6f} eV/atom")
print(f"  ΔE/atom = {energy_diff_per_atom * 1000:+.4f} meV/atom")

# Convertir a kJ/mol
kJ_per_mol = energy_diff_per_atom * 96.485  # 1 eV/atom = 96.485 kJ/mol
print(f"  ΔE/atom = {kJ_per_mol:+.4f} kJ/mol")

# Determinar cuál es más estable
if abs(energy_diff) < 1e-4:
    stability = "Las fases tienen ENERGÍA EQUIVALENTE"
elif energy_diff < 0:
    stability = "La fase ORTHORHOMBIC es MÁS ESTABLE"
else:
    stability = "La fase MONOCLINIC es MÁS ESTABLE"

print(f"\n🎯 CONCLUSIÓN:")
print("-"*70)
print(f"  {stability}")
print(f"  Diferencia de energía: {abs(energy_diff_per_atom * 1000):.4f} meV/atom")

# Comparación de volúmenes
print(f"\n📊 VOLÚMENES FINALES:")
print("-"*70)
print(f"  Orthorhombic: {ortho['volume_final']:.6f} Ų")
print(f"  Monoclinic:   {mono['volume_final']:.6f} Ų")

vol_diff = ortho['volume_final'] - mono['volume_final']
vol_diff_pct = vol_diff / mono['volume_final'] * 100

print(f"\n  ΔV (Ortho - Mono) = {vol_diff:+.6f} Ų ({vol_diff_pct:+.4f} %)")

# Comparación de densidades
density_ortho = n_atoms / ortho['volume_final']
density_mono = n_atoms / mono['volume_final']

print(f"\n📊 DENSIDADES (átomos/Ų):")
print("-"*70)
print(f"  Orthorhombic: {density_ortho:.6f} átomos/Ų")
print(f"  Monoclinic:   {density_mono:.6f} átomos/Ų")
print(f"  Δρ = {density_ortho - density_mono:+.6f} átomos/Ų")

# Verificar transición de fase
print(f"\n📊 ANÁLISIS DE SIMETRÍA:")
print("-"*70)
print(f"  Orthorhombic inicial: {'Ortho' if ortho['is_orthorhombic_initial'] else 'Mono'} → "
      f"Final: {'Ortho' if ortho['is_orthorhombic_final'] else 'Mono'}")
print(f"  Monoclinic inicial:   {'Ortho' if mono['is_orthorhombic_initial'] else 'Mono'} → "
      f"Final: {'Ortho' if mono['is_orthorhombic_final'] else 'Mono'}")

if ortho['is_orthorhombic_final'] == mono['is_orthorhombic_final']:
    print(f"\n  ⚠️  Ambas fases convergen al mismo tipo de simetría!")
else:
    print(f"\n  ✓ Las fases mantienen simetrías diferentes")

# Guardar resumen en archivo de texto
summary_file = f"{output_dir}/mfi_phase_comparison_summary.txt"
with open(summary_file, 'w') as f:
    f.write("="*70 + "\n")
    f.write(" COMPARACIÓN DE FASES MFI: ORTHORHOMBIC vs MONOCLINIC\n")
    f.write("="*70 + "\n\n")
    f.write(f"Modelo: {model_path}\n")
    f.write(f"Presión: {pressure_gpa} GPa\n")
    f.write(f"Convergencia: {fmax} eV/Å\n\n")
    
    f.write("ENERGÍAS FINALES:\n")
    f.write(f"  Orthorhombic: {ortho['energy_final']:.6f} eV ({ortho['energy_final']/n_atoms:.6f} eV/atom)\n")
    f.write(f"  Monoclinic:   {mono['energy_final']:.6f} eV ({mono['energy_final']/n_atoms:.6f} eV/atom)\n\n")
    
    f.write("DIFERENCIA ENERGÉTICA (Ortho - Mono):\n")
    f.write(f"  ΔE       = {energy_diff:+.6f} eV\n")
    f.write(f"  ΔE/atom  = {energy_diff_per_atom:+.6f} eV/atom\n")
    f.write(f"  ΔE/atom  = {energy_diff_per_atom * 1000:+.4f} meV/atom\n")
    f.write(f"  ΔE/atom  = {kJ_per_mol:+.4f} kJ/mol\n\n")
    
    f.write(f"CONCLUSIÓN: {stability}\n\n")
    
    f.write("VOLÚMENES FINALES:\n")
    f.write(f"  Orthorhombic: {ortho['volume_final']:.6f} Ų\n")
    f.write(f"  Monoclinic:   {mono['volume_final']:.6f} Ų\n")
    f.write(f"  ΔV = {vol_diff:+.6f} Ų ({vol_diff_pct:+.4f} %)\n\n")
    
    f.write("PARÁMETROS DE CELDA FINALES:\n")
    f.write("Orthorhombic:\n")
    f.write(f"  a = {ortho['cell_final'][0]:.6f} Å\n")
    f.write(f"  b = {ortho['cell_final'][1]:.6f} Å\n")
    f.write(f"  c = {ortho['cell_final'][2]:.6f} Å\n")
    f.write(f"  α = {ortho['cell_final'][3]:.4f}°\n")
    f.write(f"  β = {ortho['cell_final'][4]:.4f}°\n")
    f.write(f"  γ = {ortho['cell_final'][5]:.4f}°\n\n")
    
    f.write("Monoclinic:\n")
    f.write(f"  a = {mono['cell_final'][0]:.6f} Å\n")
    f.write(f"  b = {mono['cell_final'][1]:.6f} Å\n")
    f.write(f"  c = {mono['cell_final'][2]:.6f} Å\n")
    f.write(f"  α = {mono['cell_final'][3]:.4f}°\n")
    f.write(f"  β = {mono['cell_final'][4]:.4f}°\n")
    f.write(f"  γ = {mono['cell_final'][5]:.4f}°\n")

print(f"\n💾 Resumen guardado: {summary_file}")

print("\n" + "="*70)
print(" ✅ ANÁLISIS COMPLETADO")
print("="*70)
print(f"\nArchivos generados en: {output_dir}/")
print("  - mfi_*_minimized.{vasp,xyz,cif}: Estructuras optimizadas")
print("  - mfi_*_minimization.{traj,log}: Trayectorias y logs")
print(f"  - {os.path.basename(summary_file)}: Resumen de resultados")
print("\n" + "="*70 + "\n")
