#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark de timestep para MD con ASE + MACE.
Protocolo (por cada timestep):
 1) Leer estructura
 2) Asignar calculadora MACE
 3) Equilibración con Langevin (NVT)
 4) Producción con MTKNPT (NPT)
 5) Guardar trayectorias y logs para análisis posterior
"""
import os
import numpy as np
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators.mace import MACECalculator
from ase.md.nose_hoover_chain import MTKNPT
from ase.md.langevin import Langevin
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
# ---------------------------
# USUARIO: parámetros
# ---------------------------
input_structure = "../structures/CONTCAR_FAU.vasp"
model_path = "../../zeolite-mh-finetuning-source.model"
device = "cuda"
default_dtype = "float64"
T = 300.0            # temperatura en K
press_au = 0.0       # presión en unidades ASE (AU)
timesteps_fs = np.arange(0.75, 1.76, 0.25)   # lista a probar (fs)
equil_time_fs = 2000.0   # tiempo de equilibración Langevin (fs)
prod_time_fs = 5000.0    # tiempo de producción MTKNPT (fs)
dump_interval_fs = 100.0  # intervalo de guardado (fs)
friction_langevin = 0.01 # coeficiente de fricción para Langevin
output_dir = "timestep_benchmark_results"

# número de pasos calculado por timestep
def fs_to_steps(dt_fs, time_fs):
    return int(max(1, round(time_fs / dt_fs)))

# ---------------------------
# Loop principal
# ---------------------------
os.makedirs(output_dir, exist_ok=True)

for dt in timesteps_fs:
    print(f"\n=== Probando timestep = {dt} fs ===")
    
    # Leer estructura
    atoms = read(input_structure)
    
    # Velocidades iniciales Maxwell-Boltzmann
    MaxwellBoltzmannDistribution(atoms, temperature_K=T)

    calc = MACECalculator(
        model_paths=model_path,
        device=device,
        default_dtype=default_dtype,
        enable_cueq=True
    )
    atoms.calc = calc

    # Parámetros temporales
    tdamp = 100.0 * dt   # fs
    pdamp = 1000.0 * dt  # fs

    # Número de pasos
    nsteps_equil = fs_to_steps(dt, equil_time_fs)
    nsteps_prod = fs_to_steps(dt, prod_time_fs)
    dump_interval_steps = max(1, int(round(dump_interval_fs / dt)))

    # ===================================
    # 1) EQUILIBRACIÓN con Langevin (NVT)
    # ===================================
    equil_log = os.path.join(output_dir, f"equil_dt{dt:.3f}fs.log")
    traj_equil = os.path.join(output_dir, f"equil_dt{dt:.3f}fs.traj")
    print(f"  Equilibración Langevin: {equil_time_fs} fs -> {nsteps_equil} pasos")
    
    try:
        dyn_equil = Langevin(
            atoms,
            timestep=dt,
            temperature_K=T,
            friction=friction_langevin,
            logfile=equil_log,
            trajectory=traj_equil,
            loginterval=dump_interval_steps
        )
        dyn_equil.run(nsteps_equil)
    except Exception as e:
        print(f"  Error en equilibración: {e}")
        continue

    # ===================================
    # 2) PRODUCCIÓN con MTKNPT (NPT)
    # ===================================
    prod_log = os.path.join(output_dir, f"prod_dt{dt:.3f}fs.log")
    traj_prod = os.path.join(output_dir, f"prod_dt{dt:.3f}fs.traj")
    print(f"  Producción MTKNPT: {prod_time_fs} fs -> {nsteps_prod} pasos")
    
    try:
        dyn_prod = MTKNPT(
            atoms,
            timestep=dt,
            temperature_K=T,
            pressure_au=press_au,
            tdamp=tdamp,
            pdamp=pdamp,
            logfile=prod_log,
            trajectory=traj_prod,
            loginterval=dump_interval_steps
        )
        dyn_prod.run(nsteps_prod)
    except Exception as e:
        print(f"  Error en producción: {e}")
        continue

    print(f"  ✓ Timestep {dt} fs completado")

print("\n✓ Benchmark terminado. Resultados en:", output_dir)
print("\nArchivos generados por timestep:")
print("  - equil_dt{dt}fs.log/traj  : Equilibración Langevin")
print("  - prod_dt{dt}fs.log/traj   : Producción MTKNPT")

