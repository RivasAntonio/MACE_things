"""Plot per-atom energy against per-atom free energy from an XYZ trajectory."""

import matplotlib.pyplot as plt
from ase.io import read
import sys
import os

def plot_energies(filename):
    print(f"Leyendo archivo: {filename} ...")
    
    # Leemos todas las configuraciones del archivo (index=':')
    try:
        frames = read(filename, index=':')
    except Exception as e:
        print(f"Error leyendo el archivo con ASE: {e}")
        return

    energies_per_atom = []
    free_energies_per_atom = []
    
    count = 0
    
    for atoms in frames:
        # Obtenemos el número de átomos de la configuración actual
        n_atoms = len(atoms)
        
        # Extraemos la información del diccionario .info que ASE crea automáticamente
        # Usamos .get() para evitar errores si alguna config no tiene la key
        energy = atoms.info.get('energy')
        free_energy = atoms.info.get('free_energy')
        
        if energy is not None and free_energy is not None:
            # Normalizamos por el número de átomos
            e_pa = energy / n_atoms
            fe_pa = free_energy / n_atoms
            
            energies_per_atom.append(e_pa)
            free_energies_per_atom.append(fe_pa)
            count += 1
        else:
            print(f"Advertencia: Configuración sin 'energy' o 'free_energy' saltada.")

    if count == 0:
        print("No se encontraron datos válidos para graficar.")
        return

    print(f"Graficando {count} configuraciones...")

    # --- Configuración del Gráfico ---
    plt.figure(figsize=(8, 6))
    
    # Gráfico de dispersión (Scatter plot)
    plt.scatter(energies_per_atom, free_energies_per_atom, 
                alpha=0.7, c='blue', edgecolors='k', s=50, label='Configuraciones')
    
    # Línea de identidad (y=x) para referencia visual
    min_val = min(min(energies_per_atom), min(free_energies_per_atom))
    max_val = max(max(energies_per_atom), max(free_energies_per_atom))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x', alpha=0.5)

    plt.xlabel('Energía / átomo (eV)')
    plt.ylabel('Energía Libre / átomo (eV)')
    plt.title(f'Energía vs Energía Libre (Normalizada)\nArchivo: {os.path.basename(filename)}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Uso: python script.py archivo.xyz
    if len(sys.argv) < 2:
        print("Uso: python script.py <archivo.xyz>")
    else:
        plot_energies(sys.argv[1])