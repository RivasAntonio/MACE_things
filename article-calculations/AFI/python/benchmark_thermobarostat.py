#!/usr/bin/env python3
"""
Benchmark para comparar diferentes termobarostatos con parámetros fijos

Este script compara diferentes implementaciones de termobarostatos NPT
usando constantes fijas optimizadas: tdamp=100*timestep, pdamp=1000*timestep

Termobarostatos probados:
  - NPTBerendsen (Berendsen)
  - Inhomogeneous_NPTBerendsen (Berendsen anisotrópico)
  - IsotropicMTKNPT (Martyna-Tobias-Klein isotrópico)
  - MTKNPT (Martyna-Tobias-Klein completo)
  - LangevinBAOAB (Langevin BAOAB)
  - MelchionnaNPT (Melchionna)

Análisis:
  - Estabilidad de temperatura y presión
  - Convergencia de propiedades termodinámicas
  - Performance comparativa
  - Comportamiento de la celda unidad
"""

from ase.io import read, write

from ase.md.nptberendsen import NPTBerendsen, Inhomogeneous_NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nose_hoover_chain import IsotropicMTKNPT, MTKNPT, NoseHooverChainNVT
#from ase.md.langevinbaoab import LangevinBAOAB
#from ase.md.
#from ase.md.melchionna import MelchionnaNPT
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
input_structure = "../structures/AFI_MS_linear.xyz"

# Parámetros MD fijos
temperature_K = 800.0          # Temperatura alta para test de estabilidad
pressure_GPa = 0.0             # Presión en GPa
timestep_fs = 0.25             # Paso de tiempo en fs
simulation_ps = 10.0           # Tiempo de simulación por benchmark
dump_interval = 50             # Guardar cada N pasos

# Parámetros de termostato y barostato (FIJOS)
tdamp_fs = 100 * timestep_fs   # tdamp = 100 * timestep = 25 fs
pdamp_fs = 1000 * timestep_fs  # pdamp = 1000 * timestep = 250 fs

# Directorio de salida
output_dir = "benchmark_thermobarostats"
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# DEFINICIÓN DE TERMOBAROSTATOS A PROBAR
# ============================================================================

thermobarostats = [
    {
        'name': 'NPTBerendsen',
        'class': NPTBerendsen,
        'params': {
            'taut': tdamp_fs * units.fs,
            'taup': pdamp_fs * units.fs,
            'pressure_au': pressure_GPa * units.GPa,
            'compressibility_au': 4.57e-5 / units.GPa  # Compresibilidad típica de zeolitas
        },
        'requires_upper_triangular': False
    },
    {
        'name': 'Inhomogeneous_NPTBerendsen',
        'class': Inhomogeneous_NPTBerendsen,
        'params': {
            'taut': tdamp_fs * units.fs,
            'taup': pdamp_fs * units.fs,
            'pressure_au': pressure_GPa * units.GPa,
            'compressibility_au': 4.57e-5 / units.GPa
        },
        'requires_upper_triangular': False
    },
    {
        'name': 'IsotropicMTKNPT',
        'class': IsotropicMTKNPT,
        'params': {
            'temperature_K': temperature_K,
            'pressure_au': pressure_GPa * units.GPa,
            'tdamp': tdamp_fs * units.fs,
            'pdamp': pdamp_fs * units.fs
        },
        'requires_upper_triangular': True
    },
    {
        'name': 'MTKNPT',
        'class': MTKNPT,
        'params': {
            'temperature_K': temperature_K,
            'pressure_au': pressure_GPa * units.GPa,
            'tdamp': tdamp_fs * units.fs,
            'pdamp': pdamp_fs * units.fs
        },
        'requires_upper_triangular': True
    },
#    {
#        'name': 'LangevinBAOAB',
#        'class': LangevinBAOAB,
#        'params': {
#            'temperature_K': temperature_K,
#            'pressure_au': pressure_GPa * units.GPa,
#            'friction': 1.0 / (tdamp_fs * units.fs),  # friction = 1/tdamp
#            'pdamp': pdamp_fs * units.fs
#        },
#        'requires_upper_triangular': False
#    },
#    {
#        'name': 'MelchionnaNPT',
#        'class': MelchionnaNPT,
#        'params': {
#            'temperature_K': temperature_K,
#            'pressure_au': pressure_GPa * units.GPa,
#            'ttime': tdamp_fs * units.fs,
#            'ptime': pdamp_fs * units.fs
#        },
#        'requires_upper_triangular': False
#    }
]

n_tests = len(thermobarostats)
n_steps = int(simulation_ps * 1000 / timestep_fs)

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

print("=" * 80)
print(" BENCHMARK: COMPARACIÓN DE TERMOBAROSTATOS NPT")
print("=" * 80)
print(f"\n📁 Directorio de salida: {output_dir}")
print(f"\n📂 Archivos de entrada:")
print(f"  Estructura: {input_structure}")
print(f"  Modelo MACE: {model_path}")
print(f"\n⚙️  Parámetros fijos:")
print(f"  Temperatura: {temperature_K} K")
print(f"  Presión: {pressure_GPa} GPa")
print(f"  Timestep: {timestep_fs} fs")
print(f"  tdamp: {tdamp_fs:.2f} fs (100 × timestep)")
print(f"  pdamp: {pdamp_fs:.2f} fs (1000 × timestep)")
print(f"  Tiempo de simulación: {simulation_ps} ps ({n_steps} pasos)")
print(f"  Intervalo de guardado: cada {dump_interval} pasos")
print(f"\n🔬 Termobarostatos a probar:")
for i, tb in enumerate(thermobarostats, 1):
    print(f"  {i}. {tb['name']}")
print(f"\n💻 Device: CUDA")
print(f"  CuEq: Activado")
print("=" * 80 + "\n")

# Leer estructura inicial
atoms_initial = read(input_structure)
n_atoms = len(atoms_initial)

print(f"📊 Estructura AFI:")
print(f"  Átomos: {n_atoms}")
print(f"  Volumen inicial: {atoms_initial.get_volume():.2f} Å³")
print(f"  Celda: a={atoms_initial.cell.cellpar()[0]:.3f} Å, "
      f"b={atoms_initial.cell.cellpar()[1]:.3f} Å, "
      f"c={atoms_initial.cell.cellpar()[2]:.3f} Å")
print(f"  Ángulos: α={atoms_initial.cell.cellpar()[3]:.2f}°, "
      f"β={atoms_initial.cell.cellpar()[4]:.2f}°, "
      f"γ={atoms_initial.cell.cellpar()[5]:.2f}°")
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
    'thermobarostat': [],
    'T_mean': [],
    'T_std': [],
    'P_mean': [],
    'P_std': [],
    'V_mean': [],
    'V_std': [],
    'V_change_pct': [],
    'E_mean': [],
    'E_std': [],
    'a_mean': [],
    'a_std': [],
    'b_mean': [],
    'b_std': [],
    'c_mean': [],
    'c_std': [],
    'time_total': [],
    'time_per_step': [],
    'time_per_atom_per_step': [],
    'steps_per_second': [],
    'stability_score': [],
    'success': []
}

# ============================================================================
# BUCLE DE BENCHMARK
# ============================================================================

print(f"\n{'=' * 80}")
print(" EJECUTANDO BENCHMARKS")
print(f"{'=' * 80}\n")

for idx, tb_config in enumerate(thermobarostats, 1):
    
    tb_name = tb_config['name']
    print(f"\n{'─' * 80}")
    print(f"  TEST {idx}/{n_tests}: {tb_name}")
    print(f"{'─' * 80}")
    
    try:
        # Copiar estructura inicial
        atoms = atoms_initial.copy()
        atoms.calc = calc
        
        # Inicializar velocidades
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
        
        # Preparar celda upper-triangular si es necesario
        if tb_config['requires_upper_triangular']:
            print(f"  🔧 Convirtiendo celda a upper-triangular...")
            cell_transpose = atoms.cell.T
            upper_triangular_cell = np.linalg.cholesky(cell_transpose @ cell_transpose.T)
            upper_triangular_cell = upper_triangular_cell.T
            atoms.set_cell(upper_triangular_cell, scale_atoms=True)
        
        # Archivos de salida
        traj_file = f"{output_dir}/{tb_name}.traj"
        
        # Arrays para recolectar datos
        temperatures = []
        pressures = []
        volumes = []
        energies = []
        cell_params = []
        
        def collect_data():
            """Recolecta datos durante la simulación"""
            temperatures.append(atoms.get_temperature())
            
            # Calcular presión (stress en GPa)
            try:
                stress = atoms.get_stress(voigt=False)  # en eV/Å³
                pressure_GPa_inst = -np.trace(stress) / 3.0 / units.GPa
                pressures.append(pressure_GPa_inst)
            except:
                pressures.append(np.nan)
            
            volumes.append(atoms.get_volume())
            energies.append(atoms.get_potential_energy())
            cell_params.append(atoms.cell.cellpar()[:3])  # a, b, c
        
        # Crear dinámica con el termobarostato correspondiente
        params = tb_config['params'].copy()
        params['temperature_K'] = temperature_K
        params['timestep'] = timestep_fs * units.fs
        params['trajectory'] = traj_file
        params['loginterval'] = dump_interval
        
        dyn = tb_config['class'](atoms, **params)
        
        # Adjuntar recolección de datos
        dyn.attach(collect_data, interval=dump_interval)
        
        equil = NoseHooverChainNVT(atoms, timestep=timestep_fs * units.fs, temperature_K=temperature_K, tdamp=tdamp_fs * units.fs)
        equil.run(int(10 * 1000 / timestep_fs))  # Equilibrar durante 10 ps
        # Medir tiempo de ejecución
        print(f"  ⏳ Ejecutando simulación...")
        start_time = time.time()
        
        # Ejecutar simulación
        dyn.run(n_steps)
        elapsed_time = time.time() - start_time
        
        # Convertir a arrays numpy
        temperatures = np.array(temperatures)
        pressures = np.array(pressures)
        volumes = np.array(volumes)
        energies = np.array(energies)
        cell_params = np.array(cell_params)
        
        # Calcular estadísticas
        T_mean = np.mean(temperatures)
        T_std = np.std(temperatures)
        P_mean = np.nanmean(pressures)
        P_std = np.nanstd(pressures)
        V_mean = np.mean(volumes)
        V_std = np.std(volumes)
        V_change_pct = (V_mean - atoms_initial.get_volume()) / atoms_initial.get_volume() * 100
        E_mean = np.mean(energies)
        E_std = np.std(energies)
        
        # Estadísticas de parámetros de celda
        a_mean, b_mean, c_mean = np.mean(cell_params, axis=0)
        a_std, b_std, c_std = np.std(cell_params, axis=0)
        
        # Métricas de performance
        time_per_step = elapsed_time / n_steps * 1000  # ms
        time_per_atom_per_step = elapsed_time / n_steps / n_atoms * 1e6  # µs
        steps_per_second = n_steps / elapsed_time
        
        # Calcular puntuación de estabilidad (menor es mejor)
        T_error = abs(T_mean - temperature_K) / temperature_K
        P_error = abs(P_mean - pressure_GPa) if not np.isnan(P_mean) else 10.0
        stability_score = T_error + T_std/temperature_K + P_error + P_std
        
        # Guardar resultados
        results['thermobarostat'].append(tb_name)
        results['T_mean'].append(T_mean)
        results['T_std'].append(T_std)
        results['P_mean'].append(P_mean)
        results['P_std'].append(P_std)
        results['V_mean'].append(V_mean)
        results['V_std'].append(V_std)
        results['V_change_pct'].append(V_change_pct)
        results['E_mean'].append(E_mean)
        results['E_std'].append(E_std)
        results['a_mean'].append(a_mean)
        results['a_std'].append(a_std)
        results['b_mean'].append(b_mean)
        results['b_std'].append(b_std)
        results['c_mean'].append(c_mean)
        results['c_std'].append(c_std)
        results['time_total'].append(elapsed_time)
        results['time_per_step'].append(time_per_step)
        results['time_per_atom_per_step'].append(time_per_atom_per_step)
        results['steps_per_second'].append(steps_per_second)
        results['stability_score'].append(stability_score)
        results['success'].append(True)
        
        # Imprimir resumen
        print(f"\n  ✅ Completado en {elapsed_time:.2f} s")
        print(f"  📊 Estadísticas termodinámicas:")
        print(f"     • Temperatura: {T_mean:.2f} ± {T_std:.2f} K (target: {temperature_K} K)")
        print(f"     • Presión: {P_mean:.4f} ± {P_std:.4f} GPa (target: {pressure_GPa} GPa)")
        print(f"     • Volumen: {V_mean:.2f} ± {V_std:.2f} Å³ ({V_change_pct:+.2f}% cambio)")
        print(f"     • Energía: {E_mean:.4f} ± {E_std:.4f} eV")
        print(f"  📏 Parámetros de celda:")
        print(f"     • a: {a_mean:.3f} ± {a_std:.3f} Å")
        print(f"     • b: {b_mean:.3f} ± {b_std:.3f} Å")
        print(f"     • c: {c_mean:.3f} ± {c_std:.3f} Å")
        print(f"  ⚡ Performance:")
        print(f"     • Tiempo por paso: {time_per_step:.3f} ms")
        print(f"     • Tiempo por átomo·paso: {time_per_atom_per_step:.3f} µs")
        print(f"     • Pasos por segundo: {steps_per_second:.2f}")
        print(f"  🎯 Estabilidad: {stability_score:.6f} (menor es mejor)")
        
    except Exception as e:
        print(f"\n  ❌ Error en simulación: {e}")
        import traceback
        traceback.print_exc()
        
        # Guardar NaN para este caso
        results['thermobarostat'].append(tb_name)
        for key in ['T_mean', 'T_std', 'P_mean', 'P_std', 'V_mean', 'V_std', 
                    'V_change_pct', 'E_mean', 'E_std', 'a_mean', 'a_std',
                    'b_mean', 'b_std', 'c_mean', 'c_std', 'time_total', 
                    'time_per_step', 'time_per_atom_per_step', 
                    'steps_per_second', 'stability_score']:
            results[key].append(np.nan)
        results['success'].append(False)

# ============================================================================
# GUARDAR RESULTADOS
# ============================================================================

print(f"\n{'=' * 80}")
print(" GUARDANDO RESULTADOS")
print(f"{'=' * 80}\n")

# Guardar en archivo de texto
results_file = f"{output_dir}/benchmark_results.txt"
with open(results_file, 'w') as f:
    f.write("# Benchmark comparativo de termobarostatos NPT\n")
    f.write(f"# Estructura: {input_structure}\n")
    f.write(f"# Temperatura: {temperature_K} K\n")
    f.write(f"# Presión: {pressure_GPa} GPa\n")
    f.write(f"# Timestep: {timestep_fs} fs\n")
    f.write(f"# tdamp: {tdamp_fs:.2f} fs (100 × timestep)\n")
    f.write(f"# pdamp: {pdamp_fs:.2f} fs (1000 × timestep)\n")
    f.write(f"# Simulación: {simulation_ps} ps ({n_steps} pasos)\n")
    f.write(f"# Átomos: {n_atoms}\n")
    f.write("\n")
    
    # Tabla principal
    f.write("=" * 140 + "\n")
    f.write("RESULTADOS TERMODINÁMICOS\n")
    f.write("=" * 140 + "\n")
    header = (f"{'Termobarostat':<30} {'T_mean(K)':<12} {'T_std(K)':<10} "
              f"{'P_mean(GPa)':<13} {'P_std(GPa)':<11} {'V_mean(Å³)':<13} "
              f"{'V_std(Å³)':<10} {'ΔV(%)':<10} {'E_mean(eV)':<13} {'E_std(eV)':<11}\n")
    f.write(header)
    f.write("-" * 140 + "\n")
    
    for i in range(len(results['thermobarostat'])):
        if results['success'][i]:
            line = (f"{results['thermobarostat'][i]:<30} "
                    f"{results['T_mean'][i]:<12.2f} {results['T_std'][i]:<10.2f} "
                    f"{results['P_mean'][i]:<13.4f} {results['P_std'][i]:<11.4f} "
                    f"{results['V_mean'][i]:<13.2f} {results['V_std'][i]:<10.2f} "
                    f"{results['V_change_pct'][i]:<10.2f} "
                    f"{results['E_mean'][i]:<13.4f} {results['E_std'][i]:<11.4f}\n")
            f.write(line)
        else:
            f.write(f"{results['thermobarostat'][i]:<30} FAILED\n")
    
    f.write("\n")
    f.write("=" * 140 + "\n")
    f.write("PARÁMETROS DE CELDA\n")
    f.write("=" * 140 + "\n")
    header2 = (f"{'Termobarostat':<30} {'a_mean(Å)':<12} {'a_std(Å)':<10} "
               f"{'b_mean(Å)':<12} {'b_std(Å)':<10} {'c_mean(Å)':<12} {'c_std(Å)':<10}\n")
    f.write(header2)
    f.write("-" * 140 + "\n")
    
    for i in range(len(results['thermobarostat'])):
        if results['success'][i]:
            line = (f"{results['thermobarostat'][i]:<30} "
                    f"{results['a_mean'][i]:<12.3f} {results['a_std'][i]:<10.3f} "
                    f"{results['b_mean'][i]:<12.3f} {results['b_std'][i]:<10.3f} "
                    f"{results['c_mean'][i]:<12.3f} {results['c_std'][i]:<10.3f}\n")
            f.write(line)
    
    f.write("\n")
    f.write("=" * 140 + "\n")
    f.write("PERFORMANCE Y ESTABILIDAD\n")
    f.write("=" * 140 + "\n")
    header3 = (f"{'Termobarostat':<30} {'Time(s)':<10} {'ms/step':<10} "
               f"{'µs/atom·step':<14} {'step/s':<10} {'Stability':<12}\n")
    f.write(header3)
    f.write("-" * 140 + "\n")
    
    for i in range(len(results['thermobarostat'])):
        if results['success'][i]:
            line = (f"{results['thermobarostat'][i]:<30} "
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
print(" ANÁLISIS COMPARATIVO")
print(f"{'=' * 80}\n")

# Filtrar solo resultados exitosos
success_mask = np.array(results['success'])
if np.any(success_mask):
    
    stability_scores = np.array([results['stability_score'][i] 
                                 for i in range(len(results['success'])) 
                                 if results['success'][i]])
    success_names = [results['thermobarostat'][i] 
                     for i in range(len(results['success'])) 
                     if results['success'][i]]
    
    # Encontrar mejor configuración (menor stability_score)
    best_idx_in_success = np.argmin(stability_scores)
    best_name = success_names[best_idx_in_success]
    best_idx = results['thermobarostat'].index(best_name)
    
    print("🏆 TERMOBAROSTATO MÁS ESTABLE:")
    print(f"   • Método: {best_name}")
    print(f"   • Temperatura: {results['T_mean'][best_idx]:.2f} ± {results['T_std'][best_idx]:.2f} K")
    print(f"   • Presión: {results['P_mean'][best_idx]:.4f} ± {results['P_std'][best_idx]:.4f} GPa")
    print(f"   • Volumen: {results['V_mean'][best_idx]:.2f} ± {results['V_std'][best_idx]:.2f} Å³ "
          f"({results['V_change_pct'][best_idx]:+.2f}%)")
    print(f"   • Performance: {results['time_per_atom_per_step'][best_idx]:.3f} µs/átomo·paso")
    print(f"   • Stability score: {results['stability_score'][best_idx]:.6f}")
    
    # Encontrar mejor performance
    perf_scores = np.array([results['time_per_atom_per_step'][i] 
                            for i in range(len(results['success'])) 
                            if results['success'][i]])
    fastest_idx_in_success = np.argmin(perf_scores)
    fastest_name = success_names[fastest_idx_in_success]
    fastest_idx = results['thermobarostat'].index(fastest_name)
    
    print(f"\n⚡ TERMOBAROSTATO MÁS RÁPIDO:")
    print(f"   • Método: {fastest_name}")
    print(f"   • Performance: {results['time_per_atom_per_step'][fastest_idx]:.3f} µs/átomo·paso")
    print(f"   • Temperatura: {results['T_mean'][fastest_idx]:.2f} ± {results['T_std'][fastest_idx]:.2f} K")
    print(f"   • Stability score: {results['stability_score'][fastest_idx]:.6f}")
    
    # Ranking por estabilidad
    print(f"\n📊 RANKING POR ESTABILIDAD:")
    print(f"{'Rank':<6} {'Termobarostat':<30} {'T_err(%)':<10} {'P_err(GPa)':<12} "
          f"{'µs/atom·step':<14} {'Stability':<12}")
    print("-" * 84)
    
    sorted_indices = np.argsort(stability_scores)
    for rank, idx_in_success in enumerate(sorted_indices, 1):
        name = success_names[idx_in_success]
        idx = results['thermobarostat'].index(name)
        T_error = abs(results['T_mean'][idx] - temperature_K) / temperature_K * 100
        P_error = abs(results['P_mean'][idx] - pressure_GPa)
        print(f"{rank:<6} {name:<30} {T_error:<10.2f} {P_error:<12.4f} "
              f"{results['time_per_atom_per_step'][idx]:<14.3f} "
              f"{results['stability_score'][idx]:<12.6f}")
    
    # Análisis de volumen
    print(f"\n📦 ANÁLISIS DE VOLUMEN:")
    print(f"{'Termobarostat':<30} {'V_inicial(Å³)':<15} {'V_final(Å³)':<15} "
          f"{'ΔV(%)':<10} {'V_std(Å³)':<12}")
    print("-" * 82)
    
    for i in range(len(results['thermobarostat'])):
        if results['success'][i]:
            print(f"{results['thermobarostat'][i]:<30} "
                  f"{atoms_initial.get_volume():<15.2f} "
                  f"{results['V_mean'][i]:<15.2f} "
                  f"{results['V_change_pct'][i]:<10.2f} "
                  f"{results['V_std'][i]:<12.2f}")
    
    print(f"\n{'=' * 80}")
    print(" ✅ BENCHMARK COMPLETADO")
    print(f"{'=' * 80}\n")
    
    print(f"📁 Archivos generados en: {output_dir}/")
    print(f"   • benchmark_results.txt - Tabla completa de resultados")
    print(f"   • benchmark_results.npz - Datos para análisis adicional")
    print(f"   • <termobarostat>.traj - Trayectorias de cada simulación")
    
    print(f"\n💡 RECOMENDACIONES:")
    print(f"\n   Para ESTABILIDAD máxima:")
    print(f"     → Usar: {best_name}")
    print(f"     → Stability score: {results['stability_score'][best_idx]:.6f}")
    
    print(f"\n   Para PERFORMANCE máxima:")
    print(f"     → Usar: {fastest_name}")
    print(f"     → Tiempo: {results['time_per_atom_per_step'][fastest_idx]:.3f} µs/átomo·paso")
    
    print(f"\n   Parámetros usados (optimizados):")
    print(f"     → tdamp = {tdamp_fs:.2f} fs (100 × timestep)")
    print(f"     → pdamp = {pdamp_fs:.2f} fs (1000 × timestep)")
    
else:
    print("❌ No se obtuvieron resultados válidos para ningún termobarostato")

# Resumen de éxitos/fallos
n_success = np.sum(success_mask)
n_failed = len(success_mask) - n_success
print(f"\n📈 Resumen de ejecución:")
print(f"   • Exitosos: {n_success}/{len(success_mask)}")
print(f"   • Fallidos: {n_failed}/{len(success_mask)}")

if n_failed > 0:
    print(f"\n⚠️  Termobarostatos que fallaron:")
    for i, name in enumerate(results['thermobarostat']):
        if not results['success'][i]:
            print(f"   • {name}")

print(f"\n{'=' * 80}\n")

