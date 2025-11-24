# mfi_compare_energies.py
from ase.io import read
from ase.optimize import BFGS
from ase.optimize.precon import PreconLBFGS
from ase.filters import UnitCellFilter
from mace.calculators import MACECalculator

def minimizar(path, head="Default", device="cpu", enable_cueq=False):
    atoms = read(path)
    atoms.calc = MACECalculator(model_paths=["../../zeolite-mh-finetuning.model"], device=device, default_dtype="float32", head=head, enable_cueq=enable_cueq)
    print(f'Primera energía ({path}) con head {head}, device={device}, cueq={enable_cueq}: {atoms.get_potential_energy():.6f} eV')
    
    # Minimización con BFGS
    atoms_bfgs = atoms.copy()
    atoms_bfgs.calc = MACECalculator(model_paths=["../../zeolite-mh-finetuning.model"], device=device, default_dtype="float32", head=head, enable_cueq=enable_cueq)
    opt_bfgs = BFGS(UnitCellFilter(atoms_bfgs))
    print(f"Usando BFGS para {path} con head {head} (device={device}, cueq={enable_cueq})")
    opt_bfgs.run(fmax=0.01)
    energy_bfgs = atoms_bfgs.get_potential_energy()
    print(f"Energía BFGS ({path}) con head {head} (device={device}, cueq={enable_cueq}): {energy_bfgs:.6f} eV")
    
    # Minimización con PreconLBFGS
    atoms_precon = atoms.copy()
    atoms_precon.calc = MACECalculator(model_paths=["../../zeolite-mh-finetuning.model"], device=device, default_dtype="float32", head=head, enable_cueq=enable_cueq)
    opt_precon = PreconLBFGS(atoms=atoms_precon, variable_cell=True)
    print(f"Usando PreconLBFGS para {path} con head {head} (device={device}, cueq={enable_cueq})")
    opt_precon.run(fmax=0.01)
    energy_precon = atoms_precon.get_potential_energy()
    print(f"Energía PreconLBFGS ({path}) con head {head} (device={device}, cueq={enable_cueq}): {energy_precon:.6f} eV")
    
    print(f"Diferencia BFGS-PreconLBFGS ({path}) con head {head} (device={device}, cueq={enable_cueq}): {abs(energy_bfgs - energy_precon):.6f} eV")
    print("-" * 50)
    
    return energy_bfgs, energy_precon

# Comparación de estructuras con diferentes heads y configuraciones de device/cueq
heads = ["Default", "pt_head"]
configs = [
    {"device": "cuda", "enable_cueq": True, "name": "CUDA_cueq"},
    {"device": "cpu", "enable_cueq": False, "name": "CPU_no_cueq"}
]
results = {}

for config in configs:
    device = config["device"]
    enable_cueq = config["enable_cueq"]
    config_name = config["name"]
    
    results[config_name] = {}
    
    for head in heads:
        print("\n" * 3)
        print("="*10 + f"HEAD: {head} | DEVICE: {device} | CUEQ: {enable_cueq}" + "="*10)
        
        print(f"\n--- Estructura Orthorhombic ---")
        energy_ortho_bfgs, energy_ortho_precon = minimizar("../structures/CONTCAR_MFI_orthorombic.vasp", head=head, device=device, enable_cueq=enable_cueq)
        
        print(f"\n--- Estructura Monoclinic ---")
        energy_mono_bfgs, energy_mono_precon = minimizar("../structures/CONTCAR_MFI_monoclinic.vasp", head=head, device=device, enable_cueq=enable_cueq)
        
        results[config_name][head] = {
            'ortho_bfgs': energy_ortho_bfgs,
            'ortho_precon': energy_ortho_precon,
            'mono_bfgs': energy_mono_bfgs,
            'mono_precon': energy_mono_precon
        }

print("\n" + "="*100)
print("RESUMEN COMPLETO DE RESULTADOS")
print("="*100)

for config_name in results:
    print(f"\n{'='*20} CONFIGURACIÓN: {config_name} {'='*20}")
    for head in heads:
        print(f"\n--- HEAD: {head} ---")
        print(f"Orthorhombic - BFGS: {results[config_name][head]['ortho_bfgs']:.6f} eV")
        print(f"Orthorhombic - PreconLBFGS: {results[config_name][head]['ortho_precon']:.6f} eV")
        print(f"Monoclinic - BFGS: {results[config_name][head]['mono_bfgs']:.6f} eV")
        print(f"Monoclinic - PreconLBFGS: {results[config_name][head]['mono_precon']:.6f} eV")
        
        print(f"\nDiferencias entre estructuras (head {head}):")
        print(f"BFGS: Ortho-Mono = {results[config_name][head]['ortho_bfgs'] - results[config_name][head]['mono_bfgs']:.6f} eV")
        print(f"PreconLBFGS: Ortho-Mono = {results[config_name][head]['ortho_precon'] - results[config_name][head]['mono_precon']:.6f} eV")

print(f"\n{'='*30} COMPARACIONES ENTRE CONFIGURACIONES {'='*30}")

for head in heads:
    print(f"\n--- COMPARACIÓN PARA HEAD: {head} ---")
    print("Diferencias entre CUDA_cueq y CPU_no_cueq:")
    print(f"Ortho BFGS: {results['CUDA_cueq'][head]['ortho_bfgs'] - results['CPU_no_cueq'][head]['ortho_bfgs']:.6f} eV")
    print(f"Ortho PreconLBFGS: {results['CUDA_cueq'][head]['ortho_precon'] - results['CPU_no_cueq'][head]['ortho_precon']:.6f} eV")
    print(f"Mono BFGS: {results['CUDA_cueq'][head]['mono_bfgs'] - results['CPU_no_cueq'][head]['mono_bfgs']:.6f} eV")
    print(f"Mono PreconLBFGS: {results['CUDA_cueq'][head]['mono_precon'] - results['CPU_no_cueq'][head]['mono_precon']:.6f} eV")

print(f"\n{'='*30} COMPARACIONES ENTRE HEADS {'='*30}")

for config_name in results:
    print(f"\n--- COMPARACIÓN PARA CONFIGURACIÓN: {config_name} ---")
    print("Diferencias entre Default y pt_head:")
    print(f"Ortho BFGS: {results[config_name]['Default']['ortho_bfgs'] - results[config_name]['pt_head']['ortho_bfgs']:.6f} eV")
    print(f"Ortho PreconLBFGS: {results[config_name]['Default']['ortho_precon'] - results[config_name]['pt_head']['ortho_precon']:.6f} eV")
    print(f"Mono BFGS: {results[config_name]['Default']['mono_bfgs'] - results[config_name]['pt_head']['mono_bfgs']:.6f} eV")
    print(f"Mono PreconLBFGS: {results[config_name]['Default']['mono_precon'] - results[config_name]['pt_head']['mono_precon']:.6f} eV")

