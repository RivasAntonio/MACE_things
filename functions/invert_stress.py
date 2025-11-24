from ase.io import read, write
import numpy as np
import sys

def invert_stress(input_file, output_file):
    print(f"Leyendo {input_file}...")
    
    # index=':' lee todas las configuraciones (trajectory) no solo la primera
    atoms_list = read(input_file, index=':')
    
    count = 0
    skipped = 0
    for i, at in enumerate(atoms_list):
        # Debug: mostrar qué tiene la primera estructura
        if i == 0:
            print(f"Claves en at.info: {list(at.info.keys())}")
            print(f"Claves en at.arrays: {list(at.arrays.keys())}")
        
        # Buscar stress en diferentes lugares
        stress_found = False
        
        if 'stress' in at.info:
            at.info['stress'] = -1.0 * np.array(at.info['stress'])
            stress_found = True
        elif 'MATPES_stress' in at.info:
            at.info['MATPES_stress'] = -1.0 * np.array(at.info['MATPES_stress'])
            stress_found = True
        elif 'stress' in at.arrays:
            at.arrays['stress'] = -1.0 * at.arrays['stress']
            stress_found = True
        elif hasattr(at, 'get_stress'):
            try:
                original_stress = at.get_stress()
                at.info['MATPES_stress'] = -1.0 * original_stress
                stress_found = True
            except:
                pass
        
        if stress_found:
            count += 1
        else:
            skipped += 1

    print(f"\nResumen:")
    print(f"  - Se invirtió el signo del stress en {count} configuraciones.")
    print(f"  - Se omitieron {skipped} configuraciones sin stress.")
    
    # Guardamos el nuevo archivo
    # write_results=False evita duplicar información si ya está en info
    write(output_file, atoms_list)
    print(f"Guardado en {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python invert_stress.py input.xyz output.xyz")
    else:
        invert_stress(sys.argv[1], sys.argv[2])