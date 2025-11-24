#!/usr/bin/env python3
"""
Benchmark para optimizar constantes de termostato (tdamp) y barostato (pdamp) en MTKNPT

Este script prueba diferentes combinaciones de tdamp y pdamp para encontrar
la configuración más estable y eficiente para simulaciones NPT de AFI.

Análisis:
  - Estabilidad de temperatura y presión
  - Performance (tiempo por paso por átomo)
  - Convergencia de propiedades termodinámicas
"""

from ase.io import read, write
from ase.md.nose_hoover_chain import MTKNPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from mace.calculators import MACECalculator
import numpy as np
import time
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

# Parámetros MD fijos
temperature_K = 800.0          # Temperatura alta para test de estabilidad
pressure_GPa = 0.0             # Presión en GPa
timestep_fs = 0.25              # Paso de tiempo en fs
simulation_ps = 10.0           # Tiempo de simulación por benchmark (corto)
dump_interval = 10             # Guardar cada N pasos

# Directorio de salida
output_dir = "benchmark_npt_constants_2"
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# MATRIZ DE PARÁMETROS A PROBAR
# ============================================================================

tdamp_values = [50.0, 75.0, 100.0, 150.0]  # Enfócate en valores altos
pdamp_values = [200.0, 250.0, 300.0, 400.0, 500.0]  # Rango moderado


# Generar todas las combinaciones
test_cases = []
for tdamp in tdamp_values:
    for pdamp in pdamp_values:
        test_cases.append((tdamp, pdamp))

n_tests = len(test_cases)
n_steps = int(simulation_ps * 1000 / timestep_fs)

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

print("=" * 80)
print(" BENCHMARK: CONSTANTES DE TERMOSTATO Y BAROSTATO MTKNPT")
print("=" * 80)
print(f"\n📁 Directorio de salida: {output_dir}")
print(f"\n📂 Archivos de entrada:")
print(f"  Estructura: {input_structure}")
print(f"  Modelo MACE: {model_path}")
print(f"\n⚙️  Parámetros fijos:")
print(f"  Temperatura: {temperature_K} K")
print(f"  Presión: {pressure_GPa} GPa")
print(f"  Timestep: {timestep_fs} fs")
print(f"  Tiempo de simulación: {simulation_ps} ps ({n_steps} pasos)")
print(f"  Intervalo de guardado: cada {dump_interval} pasos")
print(f"\n🔬 Matriz de parámetros:")
print(f"  tdamp (termostato): {tdamp_values} fs")
print(f"  pdamp (barostato): {pdamp_values} fs")
print(f"  Total de combinaciones: {n_tests}")
print(f"\n💻 Device: CUDA")
print(f"  CuEq: Activado")
print("=" * 80 + "\n")

# Leer estructura inicial
atoms_initial = read(input_structure)
n_atoms = len(atoms_initial)

print(f"📊 Estructura AFI:")
print(f"  Átomos: {n_atoms}")
print(f"  Volumen inicial: {atoms_initial.get_volume():.2f} Å³")
print(f"  Celda: {atoms_initial.cell.cellpar()[:3]}")
print(f"  Ángulos: {atoms_initial.cell.cellpar()[3:]}")
print()

# Configurar calculador MACE (una sola vez)
calc = MACECalculator(
    model_paths=model_path,
    device="cuda",
    default_dtype="float64",
    enable_cueq=True
)

# ============================================================================
# ALMACENAMIENTO DE RESULTADOS
# ============================================================================

results = {
    'tdamp': [],
    'pdamp': [],
    'T_mean': [],
    'T_std': [],
    'P_mean': [],
    'P_std': [],
    'V_mean': [],
    'V_std': [],
    'E_mean': [],
    'E_std': [],
    'time_total': [],
    'time_per_step': [],
    'time_per_atom_per_step': [],
    'steps_per_second': [],
    'stability_score': []
}

# ============================================================================
# BUCLE DE BENCHMARK
# ============================================================================

print(f"\n{'=' * 80}")
print(" EJECUTANDO BENCHMARKS")
print(f"{'=' * 80}\n")

for idx, (tdamp, pdamp) in enumerate(test_cases, 1):
    
    print(f"\n{'─' * 80}")
    print(f"  TEST {idx}/{n_tests}: tdamp={tdamp} fs, pdamp={pdamp} fs")
    print(f"{'─' * 80}")
    
    # Copiar estructura inicial
    atoms = atoms_initial.copy()
    atoms.calc = calc
    
    # Inicializar velocidades
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    
    # Preparar celda upper-triangular (necesario para MTKNPT)
    cell_transpose = atoms.cell.T
    upper_triangular_cell = np.linalg.cholesky(cell_transpose @ cell_transpose.T)
    upper_triangular_cell = upper_triangular_cell.T
    atoms.set_cell(upper_triangular_cell, scale_atoms=True)
    
    # Archivos de salida
    log_file = f"{output_dir}/npt_tdamp{int(tdamp)}_pdamp{int(pdamp)}.log"
    traj_file = f"{output_dir}/npt_tdamp{int(tdamp)}_pdamp{int(pdamp)}.traj"
    
    # Arrays para recolectar datos
    temperatures = []
    pressures = []
    volumes = []
    energies = []
    
    def collect_data():
        """Recolecta datos durante la simulación"""
        temperatures.append(atoms.get_temperature())
        
        # Calcular presión (stress en GPa)
        try:
            stress = atoms.get_stress(voigt=False)  # en eV/Å³
            pressure_GPa = -np.trace(stress) / 3.0 / units.GPa
            pressures.append(pressure_GPa)
        except:
            pressures.append(np.nan)
        
        volumes.append(atoms.get_volume())
        energies.append(atoms.get_potential_energy())
    
    # Crear dinámica MTKNPT
    dyn = MTKNPT(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_K,
        pressure_au=pressure_GPa * units.GPa,
        tdamp=tdamp * units.fs,
        pdamp=pdamp * units.fs,
        logfile=log_file,
        trajectory=traj_file,
        loginterval=dump_interval
    )
    
    # Adjuntar recolección de datos
    dyn.attach(collect_data, interval=dump_interval)
    
    # Medir tiempo de ejecución
    start_time = time.time()
    
    try:
        # Ejecutar simulación
        dyn.run(n_steps)
        elapsed_time = time.time() - start_time
        
        # Convertir a arrays numpy
        temperatures = np.array(temperatures)
        pressures = np.array(pressures)
        volumes = np.array(volumes)
        energies = np.array(energies)
        
        # Calcular estadísticas
        T_mean = np.mean(temperatures)
        T_std = np.std(temperatures)
        P_mean = np.nanmean(pressures)
        P_std = np.nanstd(pressures)
        V_mean = np.mean(volumes)
        V_std = np.std(volumes)
        E_mean = np.mean(energies)
        E_std = np.std(energies)
        
        # Métricas de performance
        time_per_step = elapsed_time / n_steps * 1000  # ms
        time_per_atom_per_step = elapsed_time / n_steps / n_atoms * 1e6  # µs
        steps_per_second = n_steps / elapsed_time
        
        # Calcular puntuación de estabilidad (menor es mejor)
        # Penaliza desviaciones de temperatura y presión objetivo
        T_error = abs(T_mean - temperature_K) / temperature_K
        P_error = abs(P_mean - pressure_GPa) if not np.isnan(P_mean) else 10.0
        stability_score = T_error + T_std/temperature_K + P_error + P_std
        
        # Guardar resultados
        results['tdamp'].append(tdamp)
        results['pdamp'].append(pdamp)
        results['T_mean'].append(T_mean)
        results['T_std'].append(T_std)
        results['P_mean'].append(P_mean)
        results['P_std'].append(P_std)
        results['V_mean'].append(V_mean)
        results['V_std'].append(V_std)
        results['E_mean'].append(E_mean)
        results['E_std'].append(E_std)
        results['time_total'].append(elapsed_time)
        results['time_per_step'].append(time_per_step)
        results['time_per_atom_per_step'].append(time_per_atom_per_step)
        results['steps_per_second'].append(steps_per_second)
        results['stability_score'].append(stability_score)
        
        # Imprimir resumen
        print(f"\n  ✅ Completado en {elapsed_time:.2f} s")
        print(f"  📊 Estadísticas:")
        print(f"     • Temperatura: {T_mean:.2f} ± {T_std:.2f} K (target: {temperature_K} K)")
        print(f"     • Presión: {P_mean:.4f} ± {P_std:.4f} GPa (target: {pressure_GPa} GPa)")
        print(f"     • Volumen: {V_mean:.2f} ± {V_std:.2f} Å³. {V_std/V_mean*100:.2f}%")
        print(f"     • Energía: {E_mean:.4f} ± {E_std:.4f} eV")
        print(f"  ⚡ Performance:")
        print(f"     • Tiempo por paso: {time_per_step:.3f} ms")
        print(f"     • Tiempo por átomo·paso: {time_per_atom_per_step:.3f} µs")
        print(f"     • Pasos por segundo: {steps_per_second:.2f}")
        print(f"  🎯 Estabilidad: {stability_score:.6f} (menor es mejor)")
        
    except Exception as e:
        print(f"\n  ❌ Error en simulación: {e}")
        # Guardar NaN para este caso
        results['tdamp'].append(tdamp)
        results['pdamp'].append(pdamp)
        for key in ['T_mean', 'T_std', 'P_mean', 'P_std', 'V_mean', 'V_std', 
                    'E_mean', 'E_std', 'time_total', 'time_per_step', 
                    'time_per_atom_per_step', 'steps_per_second', 'stability_score']:
            results[key].append(np.nan)

# ============================================================================
# GUARDAR RESULTADOS
# ============================================================================

print(f"\n{'=' * 80}")
print(" GUARDANDO RESULTADOS")
print(f"{'=' * 80}\n")

# Guardar en archivo de texto
results_file = f"{output_dir}/benchmark_results.txt"
with open(results_file, 'w') as f:
    f.write("# Benchmark de constantes tdamp y pdamp para MTKNPT\n")
    f.write(f"# Estructura: {input_structure}\n")
    f.write(f"# Temperatura: {temperature_K} K\n")
    f.write(f"# Presión: {pressure_GPa} GPa\n")
    f.write(f"# Timestep: {timestep_fs} fs\n")
    f.write(f"# Simulación: {simulation_ps} ps ({n_steps} pasos)\n")
    f.write(f"# Átomos: {n_atoms}\n")
    f.write("\n")
    
    # Cabecera
    header = (f"{'tdamp(fs)':<10} {'pdamp(fs)':<10} {'T_mean(K)':<12} {'T_std(K)':<10} "
              f"{'P_mean(GPa)':<13} {'P_std(GPa)':<11} {'V_mean(A³)':<12} {'V_std(A³)':<10} "
              f"{'E_mean(eV)':<12} {'E_std(eV)':<11} {'Time(s)':<10} {'ms/step':<10} "
              f"{'µs/atom·step':<14} {'step/s':<10} {'Stability':<12}\n")
    f.write(header)
    f.write("-" * len(header) + "\n")
    
    # Datos
    for i in range(len(results['tdamp'])):
        line = (f"{results['tdamp'][i]:<10.1f} {results['pdamp'][i]:<10.1f} "
                f"{results['T_mean'][i]:<12.2f} {results['T_std'][i]:<10.2f} "
                f"{results['P_mean'][i]:<13.4f} {results['P_std'][i]:<11.4f} "
                f"{results['V_mean'][i]:<12.2f} {results['V_std'][i]:<10.2f} "
                f"{results['E_mean'][i]:<12.4f} {results['E_std'][i]:<11.4f} "
                f"{results['time_total'][i]:<10.2f} {results['time_per_step'][i]:<10.3f} "
                f"{results['time_per_atom_per_step'][i]:<14.3f} "
                f"{results['steps_per_second'][i]:<10.2f} "
                f"{results['stability_score'][i]:<12.6f}\n")
        f.write(line)

print(f"💾 Resultados guardados: {results_file}")

# Guardar en formato NumPy para análisis posterior
results_npz = f"{output_dir}/benchmark_results.npz"
np.savez_compressed(results_npz, **results)
print(f"💾 Resultados NumPy guardados: {results_npz}")

# ============================================================================
# ANÁLISIS Y RECOMENDACIONES
# ============================================================================

print(f"\n{'=' * 80}")
print(" ANÁLISIS DE RESULTADOS")
print(f"{'=' * 80}\n")

# Convertir a arrays para análisis
stability_scores = np.array(results['stability_score'])
valid_idx = ~np.isnan(stability_scores)

if np.any(valid_idx):
    # Encontrar mejor configuración (menor stability_score)
    best_idx = np.nanargmin(stability_scores)
    
    print("🏆 MEJOR CONFIGURACIÓN (mayor estabilidad):")
    print(f"   • tdamp = {results['tdamp'][best_idx]:.1f} fs")
    print(f"   • pdamp = {results['pdamp'][best_idx]:.1f} fs")
    print(f"   • Temperatura: {results['T_mean'][best_idx]:.2f} ± {results['T_std'][best_idx]:.2f} K")
    print(f"   • Presión: {results['P_mean'][best_idx]:.4f} ± {results['P_std'][best_idx]:.4f} GPa")
    print(f"   • Performance: {results['time_per_atom_per_step'][best_idx]:.3f} µs/átomo·paso")
    print(f"   • Stability score: {stability_scores[best_idx]:.6f}")
    
    # Encontrar mejor performance
    perf_scores = np.array(results['time_per_atom_per_step'])
    fastest_idx = np.nanargmin(perf_scores)
    
    print(f"\n⚡ MEJOR PERFORMANCE (más rápido):")
    print(f"   • tdamp = {results['tdamp'][fastest_idx]:.1f} fs")
    print(f"   • pdamp = {results['pdamp'][fastest_idx]:.1f} fs")
    print(f"   • Performance: {results['time_per_atom_per_step'][fastest_idx]:.3f} µs/átomo·paso")
    print(f"   • Temperatura: {results['T_mean'][fastest_idx]:.2f} ± {results['T_std'][fastest_idx]:.2f} K")
    print(f"   • Stability score: {stability_scores[fastest_idx]:.6f}")
    
    # Top 5 configuraciones más estables
    print(f"\n📊 TOP 5 CONFIGURACIONES MÁS ESTABLES:")
    print(f"{'Rank':<6} {'tdamp(fs)':<10} {'pdamp(fs)':<10} {'T_err(%)':<10} "
          f"{'P_err(GPa)':<12} {'µs/atom·step':<14} {'Stability':<12}")
    print("-" * 78)
    
    sorted_idx = np.argsort(stability_scores)
    for rank, idx in enumerate(sorted_idx[:5], 1):
        if np.isnan(stability_scores[idx]):
            continue
        T_error = abs(results['T_mean'][idx] - temperature_K) / temperature_K * 100
        P_error = abs(results['P_mean'][idx] - pressure_GPa)
        print(f"{rank:<6} {results['tdamp'][idx]:<10.1f} {results['pdamp'][idx]:<10.1f} "
              f"{T_error:<10.2f} {P_error:<12.4f} "
              f"{results['time_per_atom_per_step'][idx]:<14.3f} "
              f"{stability_scores[idx]:<12.6f}")
    
    print(f"\n{'=' * 80}")
    print(" ✅ BENCHMARK COMPLETADO")
    print(f"{'=' * 80}\n")
    
    print(f"📁 Archivos generados en: {output_dir}/")
    print(f"   • benchmark_results.txt - Tabla de resultados")
    print(f"   • benchmark_results.npz - Datos para análisis")
    print(f"   • npt_tdamp*_pdamp*.log - Logs de cada simulación")
    print(f"   • npt_tdamp*_pdamp*.traj - Trayectorias de cada simulación")
    
    print(f"\n💡 RECOMENDACIÓN:")
    print(f"   Para AFI a {temperature_K} K, usar:")
    print(f"   tdamp = {results['tdamp'][best_idx]:.1f} fs")
    print(f"   pdamp = {results['pdamp'][best_idx]:.1f} fs")
    
else:
    print("❌ No se obtuvieron resultados válidos")

print(f"\n{'=' * 80}\n")

