#!/usr/bin/env python3
"""
ionic_diffusion_calculation.py

Calcula el coeficiente de difusión iónica y la conductividad iónica a partir
de una trayectoria MD usando el desplazamiento cuadrático medio (MSD).

Uso:
    python ionic_diffusion_calculation.py <traj_file> <specie> <temperature>

Argumentos posicionales:
    traj_file      Ruta al archivo de trayectoria (formato ASE, ej. production.traj)
    specie         Símbolo del elemento a estudiar (ej. I, Br, Li, Na)
    temperature    Temperatura de la simulación en Kelvin

Ejemplo:
    python ionic_diffusion_calculation.py outputs/production.traj I 600

Salida:
    - Coeficiente de difusión D (cm²/s)
    - Conductividad iónica (mS/cm)
    - Gráfica MSD vs step por pantalla
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.diffusion.analyzer import DiffusionAnalyzer

# ================= ARGUMENTOS =================
parser = argparse.ArgumentParser(description="Cálculo de difusión iónica a partir de una trayectoria.")
parser.add_argument("--traj_file", help="Ruta al archivo de trayectoria (ej. production.traj)")
parser.add_argument("--specie", help="Especie a estudiar (ej. I, Br, Li)")
parser.add_argument("--temperature", type=float, help="Temperatura de la simulación en K")
parser.add_argument("--step_skip", type=int, default=1, help="Salto entre frames para el análisis (default: 1)")
args = parser.parse_args()

TRAJ_FILE = args.traj_file
SPECIE = args.specie
TEMP = args.temperature

# ================= 1. CARGA DE DATOS =================
print(f">>> Leyendo trayectoria: {TRAJ_FILE} ...")
print("(Esto puede tardar dependiendo del tamaño del archivo)")

ase_traj = read(TRAJ_FILE, index=f':{args.step_skip}')

if len(ase_traj) < 50:
    print("¡ADVERTENCIA! Tienes muy pocos frames. Reduce el salto en index='::X'")

print(f"Frames cargados para análisis: {len(ase_traj)}")

# Convertir a objetos de Pymatgen
print(">>> Convirtiendo a estructura Pymatgen...")
structures = [AseAtomsAdaptor.get_structure(atoms) for atoms in ase_traj]

# ================= 2. CÁLCULO DE DIFUSIÓN =================
print(f">>> Calculando MSD y Difusión para el elemento '{SPECIE}'...")

analyzer = DiffusionAnalyzer.from_structures(
    structures=structures,
    specie=SPECIE,
    temperature=TEMP,
    time_step=1.0,   # En femtosegundos (valor nominal, el eje x será en steps)
    step_skip=1,
    smoothed="max"   # Usa el rango más lineal para el ajuste
)

# ================= 3. RESULTADOS Y GRAFICACIÓN =================

# Obtener valores
D = analyzer.diffusivity  # cm^2/s
sigma = analyzer.conductivity  # mS/cm

try:
    fit_data = analyzer.diffusivity_components
    if hasattr(analyzer, 'corrected_diffusivity_components'):
        fit_data = analyzer.corrected_diffusivity_components
    r2 = fit_data.get('r2', None)
except Exception:
    r2 = None

print("\n" + "="*40)
print(f" RESULTADOS A {TEMP} K para {SPECIE}")
print("="*40)
print(f"Coeficiente de Difusión (D): {D:.4e} cm^2/s")
print(f"Conductividad Iónica     : {sigma:.2f} mS/cm")
if r2 is not None:
    print(f"Calidad del ajuste (R^2) : {r2:.4f}")
print("="*40)

# Graficar MSD con eje x en steps
msd = analyzer.msd  # Array de MSD promedio (Å²)
steps = np.arange(len(msd))

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(steps, msd, label=f"MSD {SPECIE}")
ax.set_xlabel("Step")
ax.set_ylabel("MSD (Å²)")
ax.set_title(f"MSD de {SPECIE} a {TEMP} K\nD = {D:.2e} cm$^2$/s")
ax.legend()
plt.tight_layout()
plt.show()
