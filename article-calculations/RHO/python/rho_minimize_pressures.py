#!/usr/bin/env python3
"""
Script para minimizar estructuras RHO a diferentes presiones usando ASE y MACE

Este script:
- Lee las estructuras last_frames.xyz de cada directorio dir_X_bar
- Realiza una minimización a la presión correspondiente (en bares)
- Convierte presión de bares a eV/Å³ para ASE
- Guarda las estructuras minimizadas y los resultados
"""

import os
import glob
import re
import numpy as np
from pathlib import Path
from ase.io import read, write
from ase.optimize import FIRE
from ase.filters import UnitCellFilter
from mace.calculators import MACECalculator
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cuequivariance_torch")

# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================

# Conversión de unidades: 1 bar = 1e-4 GPa, 1 GPa = 1 eV/Å³ / 160.21766208
# Por lo tanto: 1 bar = 1e-4 / 160.21766208 eV/Å³
BAR_TO_EV_ANG3 = 1e-4 / 160.21766208  # ≈ 6.2415e-7 eV/Å³

# Rutas
model_path = "../../zeolite-mh-finetuning-source.model"
structures_base_dir = "../structures"
output_dir = "outputs_pressure_minimization"

# Parámetros de minimización
fmax = 0.01  # Criterio de convergencia (eV/Å)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def extract_pressure_from_dirname(dirname):
    """
    Extrae la presión en bares del nombre del directorio.
    
    Ejemplo: 'dir_5000.0_bar' -> 5000.0
    """
    match = re.search(r'dir_(\d+\.?\d*)_bar', dirname)
    if match:
        return float(match.group(1))
    return None

def bar_to_ev_ang3(pressure_bar):
    """Convierte presión de bares a eV/Å³"""
    return pressure_bar * BAR_TO_EV_ANG3

def find_all_pressure_directories():
    """
    Encuentra todos los directorios dir_*_bar en structures/
    Retorna una lista de tuplas (directorio, presión_en_bares)
    """
    pattern = os.path.join(structures_base_dir, "dir_*_bar")
    dirs = glob.glob(pattern)
    
    pressure_dirs = []
    for d in dirs:
        dirname = os.path.basename(d)
        pressure = extract_pressure_from_dirname(dirname)
        if pressure is not None:
            pressure_dirs.append((d, pressure))
    
    # Ordenar por presión
    pressure_dirs.sort(key=lambda x: x[1])
    return pressure_dirs

# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

def main():
    print("="*80)
    print(" MINIMIZACIÓN DE ESTRUCTURAS RHO A DIFERENTES PRESIONES")
    print("="*80)
    print(f"\nModelo MACE: {model_path}")
    print(f"Directorio base: {structures_base_dir}")
    print(f"Criterio de convergencia: fmax = {fmax} eV/Å")
    print(f"\nConversión de unidades:")
    print(f"  1 bar = {BAR_TO_EV_ANG3:.6e} eV/Å³")
    print(f"  1 GPa = {1.0/160.21766208:.6e} eV/Å³")
    print("="*80 + "\n")
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Directorio de salida: {output_dir}\n")
    
    # Buscar todos los directorios de presión
    pressure_dirs = find_all_pressure_directories()
    
    if not pressure_dirs:
        print("❌ No se encontraron directorios dir_*_bar en structures/")
        return
    
    print(f"✓ Se encontraron {len(pressure_dirs)} directorios de presión:")
    for d, p in pressure_dirs:
        print(f"  - {os.path.basename(d)}: {p:.1f} bar = {bar_to_ev_ang3(p):.6e} eV/Å³")
    print()
    
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
        print(f"⚠️  Error al inicializar CUDA, usando CPU: {e}")
        calc = MACECalculator(
            model_paths=model_path,
            device="cpu",
            default_dtype="float64"
        )
        print("✓ Calculador MACE inicializado (CPU)")
    print()
    
    # Archivo de resumen
    summary_file = os.path.join(output_dir, "minimization_summary.txt")
    results = []
    
    # Procesar cada presión
    for i, (pressure_dir, pressure_bar) in enumerate(pressure_dirs, 1):
        print("="*80)
        print(f" PRESIÓN {i}/{len(pressure_dirs)}: {pressure_bar:.1f} bar")
        print("="*80)
        
        dirname = os.path.basename(pressure_dir)
        input_file = os.path.join(pressure_dir, "SIMU_final", "last_frames.xyz")
        
        # Verificar que existe el archivo
        if not os.path.exists(input_file):
            print(f"⚠️  Archivo no encontrado: {input_file}")
            print("   Saltando...\n")
            continue
        
        print(f"📂 Directorio: {dirname}")
        print(f"📄 Estructura: {input_file}")
        
        # Convertir presión a eV/Å³
        pressure_ev_ang3 = bar_to_ev_ang3(pressure_bar)
        print(f"🔢 Presión: {pressure_bar:.1f} bar = {pressure_ev_ang3:.6e} eV/Å³")
        
        try:
            # Leer estructura
            atoms = read(input_file)
            print(f"\n✓ Estructura leída:")
            print(f"  Número de átomos: {len(atoms)}")
            print(f"  Volumen inicial: {atoms.get_volume():.4f} Å³")
            print(f"  Celda inicial:")
            for j, vec in enumerate(atoms.cell):
                print(f"    a{j+1} = [{vec[0]:8.4f}, {vec[1]:8.4f}, {vec[2]:8.4f}] Å")
            
            # Asignar calculador
            atoms.calc = calc
            
            # Energía inicial
            E_initial = atoms.get_potential_energy()
            print(f"  Energía inicial: {E_initial:.6f} eV")
            print(f"  Energía por átomo: {E_initial/len(atoms):.6f} eV/atom")
            
            # Preparar minimización con presión
            print(f"\n🔄 Minimizando estructura a P = {pressure_bar:.1f} bar...")
            ucf = UnitCellFilter(atoms, scalar_pressure=pressure_ev_ang3)
            
            # Configurar optimizador
            log_file = os.path.join(output_dir, f"minimization_{dirname}.log")
            traj_file = os.path.join(output_dir, f"minimization_{dirname}.traj")
            opt = FIRE(ucf, logfile=log_file, trajectory=traj_file)
            
            # Ejecutar minimización
            opt.run(fmax=fmax)
            
            # Resultados finales
            E_final = atoms.get_potential_energy()
            V_final = atoms.get_volume()
            
            print(f"\n✓ Minimización completada:")
            print(f"  Energía final: {E_final:.6f} eV")
            print(f"  Energía por átomo: {E_final/len(atoms):.6f} eV/atom")
            print(f"  Cambio de energía: {E_final - E_initial:.6f} eV")
            print(f"  Volumen final: {V_final:.4f} Å³")
            print(f"  Cambio de volumen: {V_final - atoms.get_volume():.4f} Å³")
            print(f"  Celda final:")
            for j, vec in enumerate(atoms.cell):
                print(f"    a{j+1} = [{vec[0]:8.4f}, {vec[1]:8.4f}, {vec[2]:8.4f}] Å")
            
            # Guardar estructura minimizada en varios formatos
            output_base = os.path.join(output_dir, f"RHO_minimized_{dirname}")
            write(f"{output_base}.xyz", atoms)
            write(f"{output_base}.vasp", atoms, vasp5=True)
            
            print(f"\n💾 Estructuras guardadas:")
            print(f"  - {output_base}.xyz")
            print(f"  - {output_base}.vasp")
            print(f"  - Trayectoria: {traj_file}")
            print(f"  - Log: {log_file}")
            
            # Guardar resultados para el resumen
            results.append({
                'dirname': dirname,
                'pressure_bar': pressure_bar,
                'pressure_ev_ang3': pressure_ev_ang3,
                'n_atoms': len(atoms),
                'E_initial': E_initial,
                'E_final': E_final,
                'E_per_atom': E_final / len(atoms),
                'V_initial': atoms.get_volume(),
                'V_final': V_final,
                'success': True
            })
            
        except Exception as e:
            print(f"\n❌ Error durante la minimización: {e}")
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
        f.write("="*80 + "\n")
        f.write(" RESUMEN DE MINIMIZACIONES RHO A DIFERENTES PRESIONES\n")
        f.write("="*80 + "\n\n")
        f.write(f"Modelo MACE: {model_path}\n")
        f.write(f"Criterio de convergencia: fmax = {fmax} eV/Å\n")
        f.write(f"Conversión: 1 bar = {BAR_TO_EV_ANG3:.6e} eV/Å³\n\n")
        f.write("="*80 + "\n")
        f.write(f"{'Directorio':<25} {'P (bar)':>10} {'P (eV/Å³)':>15} {'N_atoms':>8} "
                f"{'E_final (eV)':>15} {'E/atom (eV)':>15} {'V_final (Å³)':>15}\n")
        f.write("="*80 + "\n")
        
        for res in results:
            if res['success']:
                f.write(f"{res['dirname']:<25} {res['pressure_bar']:>10.1f} "
                       f"{res['pressure_ev_ang3']:>15.6e} {res['n_atoms']:>8} "
                       f"{res['E_final']:>15.6f} {res['E_per_atom']:>15.6f} "
                       f"{res['V_final']:>15.4f}\n")
            else:
                f.write(f"{res['dirname']:<25} {res['pressure_bar']:>10.1f} "
                       f"{res['pressure_ev_ang3']:>15.6e} {'ERROR':>8} "
                       f"{'N/A':>15} {'N/A':>15} {'N/A':>15}\n")
        
        f.write("="*80 + "\n\n")
        
        # Análisis adicional
        successful = [r for r in results if r['success']]
        if successful:
            f.write("ANÁLISIS:\n")
            f.write("-"*80 + "\n")
            f.write(f"Minimizaciones exitosas: {len(successful)}/{len(results)}\n\n")
            
            # Energía vs presión
            f.write("Energía por átomo vs Presión:\n")
            for res in successful:
                f.write(f"  {res['pressure_bar']:8.1f} bar: {res['E_per_atom']:12.6f} eV/atom\n")
            f.write("\n")
            
            # Volumen vs presión
            f.write("Volumen vs Presión:\n")
            for res in successful:
                f.write(f"  {res['pressure_bar']:8.1f} bar: {res['V_final']:12.4f} Å³\n")
            f.write("\n")
    
    print(f"✓ Resumen guardado en: {summary_file}")
    print("\n" + "="*80)
    print(" PROCESO COMPLETADO")
    print("="*80)
    print(f"\nMinimizaciones exitosas: {len([r for r in results if r['success']])}/{len(results)}")
    print(f"Resultados guardados en: {output_dir}/")
    print()

if __name__ == "__main__":
    main()
