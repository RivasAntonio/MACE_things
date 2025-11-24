# mfi_compare_energies_cuda.py
import warnings
import os
from ase.io import read, write
from ase.optimize import BFGS
from ase.filters import UnitCellFilter
from mace.calculators import MACECalculator

# Suprimir warnings para una salida más limpia
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

# Configurar CuEq para evitar el warning del método
os.environ["CUEQUIVARIANCE_DEFAULT_METHOD"] = "uniform_1d"

# Crear directorio de outputs
output_dir = "outputs_cuda"
os.makedirs(output_dir, exist_ok=True)
print(f"📁 Creado directorio: {output_dir}")

def minimizar(path, head="Default"):
    atoms = read(path)
    atoms.calc = MACECalculator(model_paths=["../zeolite-mh-finetuning.model"], device="cuda", default_dtype="float32", head=head, enable_cueq=True)
    print(f'Primera energía ({path}) con head {head} (CUDA + CUEQ): {atoms.get_potential_energy():.6f} eV')
    
    # Crear nombres de archivos únicos
    structure_name = os.path.basename(path).replace('.vasp', '').replace('CONTCAR_', '')
    
    # Minimización con BFGS
    atoms_bfgs = atoms.copy()
    atoms_bfgs.calc = MACECalculator(model_paths=["../zeolite-mh-finetuning.model"], device="cuda", default_dtype="float32", head=head, enable_cueq=True)
    
    # Configurar archivos de trayectoria y log para BFGS
    traj_bfgs = f"{output_dir}/{structure_name}_{head}_BFGS_cuda.traj"
    log_bfgs = f"{output_dir}/{structure_name}_{head}_BFGS_cuda.log"
    
    opt_bfgs = BFGS(UnitCellFilter(atoms_bfgs), trajectory=traj_bfgs, logfile=log_bfgs)
    print(f"Usando BFGS para {path} con head {head} (CUDA + CUEQ)")
    print(f"  📄 Trayectoria: {traj_bfgs}")
    print(f"  📋 Log: {log_bfgs}")
    opt_bfgs.run(fmax=0.01)
    energy_bfgs = atoms_bfgs.get_potential_energy()
    print(f"Energía BFGS ({path}) con head {head} (CUDA + CUEQ): {energy_bfgs:.6f} eV")
    
    # Guardar estructura final BFGS
    final_bfgs = f"{output_dir}/{structure_name}_{head}_BFGS_final_cuda.vasp"
    write(final_bfgs, atoms_bfgs, format='vasp')
    print(f"  💾 Estructura final: {final_bfgs}")
    
    print("-" * 50)
    return energy_bfgs

# Comparación de estructuras con diferentes heads usando CUDA + CUEQ
heads = ["Default", "pt_head"]
results = {}

for head in heads:
    print("\n" * 3)
    print("="*15 + f"HEAD: {head} | CUDA + CUEQ=True" + "="*15)
    print(f"\n--- Estructura Orthorhombic con head {head} ---")
    energy_ortho_bfgs = minimizar("../structures/CONTCAR_MFI_orthorombic.vasp", head=head)
    print(f"\n--- Estructura Monoclinic con head {head} ---")
    energy_mono_bfgs = minimizar("../structures/CONTCAR_MFI_monoclinic.vasp", head=head)
    results[head] = {
        'ortho_bfgs': energy_ortho_bfgs,
        'mono_bfgs': energy_mono_bfgs
    }


print("\n" + "="*80)
print("RESUMEN COMPLETO DE RESULTADOS - CUDA + CUEQ=True")
print("="*80)

for head in heads:
    print(f"\n=== RESULTADOS CON HEAD: {head} ===")
    print(f"Orthorhombic - BFGS: {results[head]['ortho_bfgs']:.6f} eV")
    print(f"Monoclinic - BFGS: {results[head]['mono_bfgs']:.6f} eV")
    print(f"\nDiferencias entre estructuras (head {head}):")
    print(f"BFGS: Ortho-Mono = {results[head]['ortho_bfgs'] - results[head]['mono_bfgs']:.6f} eV")

print(f"\n=== COMPARACIÓN ENTRE HEADS (CUDA + CUEQ) ===")
print("Diferencias entre Default y pt_head:")
print(f"Ortho BFGS: {results['Default']['ortho_bfgs'] - results['pt_head']['ortho_bfgs']:.6f} eV")
print(f"Mono BFGS: {results['Default']['mono_bfgs'] - results['pt_head']['mono_bfgs']:.6f} eV")

# Guardar resultados en archivo de texto
txt_filename = f"{output_dir}/mfi_summary_cuda.txt"
with open(txt_filename, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write(f"RESUMEN DE RESULTADOS MFI - CUDA + CUEQ=True\n")
    f.write("=" * 80 + "\n\n")
    for head in heads:
        f.write(f"=== RESULTADOS CON HEAD: {head} ===\n")
        f.write(f"Orthorhombic - BFGS: {results[head]['ortho_bfgs']:.6f} eV\n")
        f.write(f"Monoclinic - BFGS: {results[head]['mono_bfgs']:.6f} eV\n")
        f.write(f"\nDiferencias entre estructuras (head {head}):\n")
        f.write(f"BFGS: Ortho-Mono = {results[head]['ortho_bfgs'] - results[head]['mono_bfgs']:.6f} eV\n\n")
    f.write("=== COMPARACIÓN ENTRE HEADS (CUDA + CUEQ) ===\n")
    f.write("Diferencias entre Default y pt_head:\n")
    f.write(f"Ortho BFGS: {results['Default']['ortho_bfgs'] - results['pt_head']['ortho_bfgs']:.6f} eV\n")
    f.write(f"Mono BFGS: {results['Default']['mono_bfgs'] - results['pt_head']['mono_bfgs']:.6f} eV\n")

print(f"📄 Resultados guardados en: {txt_filename}")
print(f"\n✅ Análisis completado y archivo guardado!")