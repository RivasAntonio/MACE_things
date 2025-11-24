#!/usr/bin/env python3
"""
Script para minimizar estructuras RHO a diferentes presiones usando ASE, MACE y FrechetCellFilter.

Este script:
- Lee las estructuras last_frames.xyz de cada directorio dir_X_bar
- Realiza una minimización a la presión correspondiente (en bares)
- Convierte presión de bares a eV/Å³ para ASE
- Usa FrechetCellFilter para relajar celda + átomos
- Calcula la energía incluyendo el término P·V, como hace VASP
"""

import os
import glob
import re
import numpy as np
from pathlib import Path
from ase.io import read, write
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from mace.calculators import MACECalculator
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cuequivariance_torch")

# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================

# 1 bar = 1e-4 GPa, 1 GPa = 1 eV/Å³ / 160.21766208
BAR_TO_EV_ANG3 = 1e-4 / 160.21766208  # ≈ 6.2415e-7 eV/Å³

# Rutas
model_path = "../../zeolite-mh-finetuning-source.model"
structures_base_dir = "../structures"
output_dir = "outputs_pressure_minimization_fmax_0.001"

# Parámetros de minimización
fmax = 0.001  # criterio de convergencia (eV/Å)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def extract_pressure_from_dirname(dirname):
    """Extrae la presión en bares del nombre del directorio."""
    match = re.search(r'dir_(\d+\.?\d*)_bar', dirname)
    return float(match.group(1)) if match else None

def bar_to_ev_ang3(pressure_bar):
    """Convierte presión de bares a eV/Å³"""
    return pressure_bar * BAR_TO_EV_ANG3

def find_all_pressure_directories():
    """Encuentra todos los directorios dir_*_bar en structures/"""
    pattern = os.path.join(structures_base_dir, "dir_*_bar")
    dirs = glob.glob(pattern)
    pressure_dirs = []
    for d in dirs:
        pressure = extract_pressure_from_dirname(os.path.basename(d))
        if pressure is not None:
            pressure_dirs.append((d, pressure))
    return sorted(pressure_dirs, key=lambda x: x[1])

# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

def main():
    print("="*80)
    print(" MINIMIZACIÓN DE ESTRUCTURAS RHO A DIFERENTES PRESIONES (FrechetCellFilter)")
    print("="*80)
    print(f"\nModelo MACE: {model_path}")
    print(f"Directorio base: {structures_base_dir}")
    print(f"Criterio de convergencia: fmax = {fmax} eV/Å")
    print(f"\nConversión de unidades:")
    print(f"  1 bar = {BAR_TO_EV_ANG3:.6e} eV/Å³")
    print("="*80 + "\n")

    os.makedirs(output_dir, exist_ok=True)

    pressure_dirs = find_all_pressure_directories()
    if not pressure_dirs:
        print("❌ No se encontraron directorios dir_*_bar en structures/")
        return

    print(f"✓ Se encontraron {len(pressure_dirs)} directorios de presión.\n")

    # Inicializar calculador MACE
    print("🔧 Inicializando calculador MACE...")
    try:
        calc = MACECalculator(
            model_paths=model_path,
            device="cuda",
            default_dtype="float64",
            enable_cueq=True
        )
        print("✓ Calculador MACE inicializado (CUDA + CuEq habilitado)")
    except Exception as e:
        print(f"⚠️ Error CUDA, usando CPU: {e}")
        calc = MACECalculator(
            model_paths=model_path,
            device="cpu",
            default_dtype="float64"
        )
        print("✓ Calculador MACE inicializado (CPU)")
    print()

    summary_file = os.path.join(output_dir, "minimization_summary.txt")
    results = []

    for i, (pressure_dir, pressure_bar) in enumerate(pressure_dirs, 1):
        print("="*80)
        print(f" PRESIÓN {i}/{len(pressure_dirs)}: {pressure_bar:.1f} bar")
        print("="*80)

        dirname = os.path.basename(pressure_dir)
        input_file = os.path.join(pressure_dir, "SIMU_final", "last_frames.xyz")

        if not os.path.exists(input_file):
            print(f"⚠️ No se encontró {input_file}")
            continue

        pressure_ev_ang3 = bar_to_ev_ang3(pressure_bar)
        print(f"📂 {dirname} — P = {pressure_bar:.1f} bar = {pressure_ev_ang3:.6e} eV/Å³")

        try:
            atoms = read(input_file)
            atoms.calc = calc
            n_atoms = len(atoms)

            E_initial = atoms.get_potential_energy()
            V_initial = atoms.get_volume()
            E_initial_with_p = E_initial + pressure_ev_ang3 * V_initial
            print(f"  Energía inicial: {E_initial:.6f} eV (E + PV = {E_initial_with_p:.6f})")

            # === Relajación con FrechetCellFilter ===
            print(f"\n🔄 Minimizando estructura a {pressure_bar:.1f} bar...")
            filt = FrechetCellFilter(atoms, scalar_pressure=pressure_ev_ang3)

            log_file = os.path.join(output_dir, f"minimization_{dirname}.log")
            traj_file = os.path.join(output_dir, f"minimization_{dirname}.traj")
            opt = FIRE(filt, logfile=log_file, trajectory=traj_file)
            opt.run(fmax=fmax)

            # === Resultados finales ===
            E_final = atoms.get_potential_energy()
            V_final = atoms.get_volume()
            E_final_with_p = E_final + pressure_ev_ang3 * V_final

            print(f"\n✓ Minimización completada:")
            print(f"  Energía final (sin P): {E_final:.6f} eV")
            print(f"  Energía total (E + PV): {E_final_with_p:.6f} eV")
            print(f"  Energía/átomo: {E_final_with_p/n_atoms:.6f} eV/atom")
            print(f"  Volumen final: {V_final:.3f} Å³")

            # === Guardar ===
            output_base = os.path.join(output_dir, f"RHO_minimized_{dirname}")
            write(f"{output_base}.xyz", atoms)
            write(f"{output_base}.vasp", atoms, vasp5=True)

            results.append({
                'dirname': dirname,
                'pressure_bar': pressure_bar,
                'pressure_ev_ang3': pressure_ev_ang3,
                'E_final': E_final,
                'E_final_with_p': E_final_with_p,
                'E_per_atom': E_final_with_p / n_atoms,
                'V_final': V_final,
                'success': True
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({
                'dirname': dirname,
                'pressure_bar': pressure_bar,
                'pressure_ev_ang3': pressure_ev_ang3,
                'success': False,
                'error': str(e)
            })
        print()

    # ============================================================================
    # GUARDAR RESUMEN
    # ============================================================================
    print("="*80)
    print(" GUARDANDO RESUMEN DE RESULTADOS")
    print("="*80)

    with open(summary_file, 'w') as f:
        f.write("Resumen minimizaciones RHO con FrechetCellFilter\n")
        f.write("="*80 + "\n")
        f.write(f"{'Dir':<25} {'P(bar)':>10} {'E+PV (eV)':>15} {'E/atom (eV)':>15} {'Vol (Å³)':>15}\n")
        f.write("="*80 + "\n")

        for r in results:
            if r['success']:
                f.write(f"{r['dirname']:<25} {r['pressure_bar']:>10.1f} "
                        f"{r['E_final_with_p']:>15.6f} {r['E_per_atom']:>15.6f} {r['V_final']:>15.4f}\n")
            else:
                f.write(f"{r['dirname']:<25} {r['pressure_bar']:>10.1f} ERROR\n")

    print(f"\n✓ Resumen guardado en {summary_file}")
    print("="*80)
    print(" PROCESO COMPLETADO ")
    print("="*80)

if __name__ == "__main__":
    main()
