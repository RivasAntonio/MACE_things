#!/usr/bin/env python3
"""
Script para calcular ángulos O-Si-O en la dirección ĉ a partir de trayectorias MD.
Optimizado para ejecución en cluster con salida a CSV.

Uso:
    python calculate_osi_angles.py --traj trayectoria.traj --output resultados.csv
"""

import numpy as np
import pandas as pd
from ase.io.trajectory import Trajectory
from ase.geometry import get_distances
import argparse
import time
from pathlib import Path


def calculate_osi_angles_along_c(atoms, cutoff=2.0, c_threshold=0.7):
    """
    Calcula ángulos O-Si-O para enlaces Si-O alineados con la dirección ĉ.
    
    Parámetros:
    -----------
    atoms : ASE Atoms object
        Estructura a analizar
    cutoff : float
        Distancia máxima Si-O en Ångström (default: 2.0)
    c_threshold : float
        Umbral de alineación con ĉ (cos(theta) > c_threshold)
        0.7 corresponde a ~45° de desviación
    
    Retorna:
    --------
    angles : list
        Lista de ángulos O-Si-O en grados
    si_count : int
        Número de átomos de Si analizados
    bond_count : int
        Número de enlaces Si-O alineados con ĉ encontrados
    """
    # Obtener índices de átomos
    si_indices = [i for i, sym in enumerate(atoms.get_chemical_symbols()) if sym == 'Si']
    o_indices = [i for i, sym in enumerate(atoms.get_chemical_symbols()) if sym == 'O']
    
    # Vector unitario de la dirección c
    cell = atoms.get_cell()
    c_vector = cell[2] / np.linalg.norm(cell[2])
    
    angles = []
    si_with_c_bonds = 0
    total_c_bonds = 0
    
    for si_idx in si_indices:
        si_pos = atoms.positions[si_idx]
        
        # Encontrar oxígenos vecinos alineados con c
        o_neighbors = []
        o_vectors = []
        
        for o_idx in o_indices:
            o_pos = atoms.positions[o_idx]
            # Calcular distancia considerando condiciones periódicas
            distance_vec, distance = get_distances([si_pos], [o_pos], cell=cell, pbc=True)
            distance = distance[0, 0]
            distance_vec = distance_vec[0, 0]
            
            if distance < cutoff:
                # Vector Si-O normalizado
                so_vector = distance_vec / distance
                
                # Calcular alineación con c
                alignment = abs(np.dot(so_vector, c_vector))
                
                if alignment > c_threshold:
                    o_neighbors.append(o_idx)
                    o_vectors.append(distance_vec)
        
        # Contar Si con al menos un enlace en dirección c
        if len(o_neighbors) > 0:
            si_with_c_bonds += 1
            total_c_bonds += len(o_neighbors)
        
        # Calcular ángulos O-Si-O para pares de oxígenos alineados con c
        if len(o_neighbors) >= 2:
            for i in range(len(o_neighbors)):
                for j in range(i + 1, len(o_neighbors)):
                    vec1 = o_vectors[i]
                    vec2 = o_vectors[j]
                    
                    # Calcular ángulo
                    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    # Asegurar que cos_angle está en [-1, 1]
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_angle))
                    
                    angles.append(angle)
    
    return angles, si_with_c_bonds, total_c_bonds


def process_trajectory(traj_file, cutoff=2.0, c_threshold=0.7, timestep_fs=0.5, 
                       dump_interval=10, verbose=True):
    """
    Procesa una trayectoria completa y calcula ángulos O-Si-O en dirección ĉ.
    
    Parámetros:
    -----------
    traj_file : str
        Ruta al archivo de trayectoria
    cutoff : float
        Distancia máxima Si-O en Å
    c_threshold : float
        Umbral de alineación con ĉ
    timestep_fs : float
        Paso de tiempo de la simulación en fs
    dump_interval : int
        Intervalo de guardado de frames
    verbose : bool
        Imprimir progreso
        
    Retorna:
    --------
    results_df : pandas DataFrame
        DataFrame con los resultados
    """
    
    if verbose:
        print("="*70)
        print(" CÁLCULO DE ÁNGULOS O-Si-O EN DIRECCIÓN ĉ")
        print("="*70)
        print(f"\nArchivo de trayectoria: {traj_file}")
        print(f"Parámetros:")
        print(f"  Cutoff Si-O: {cutoff} Å")
        print(f"  Umbral alineación ĉ: {c_threshold} (cos θ)")
        print(f"  Timestep: {timestep_fs} fs")
        print(f"  Dump interval: {dump_interval}")
    
    # Leer trayectoria
    start_time = time.time()
    if verbose:
        print(f"\nLeyendo trayectoria...")
    
    traj = Trajectory(traj_file)
    n_frames = len(traj)
    
    if verbose:
        print(f"  Total de frames: {n_frames}")
        print(f"  Estructura: {traj[0].get_chemical_formula()}")
        print(f"  Átomos: {len(traj[0])}")
    
    # Preparar listas para almacenar resultados
    frame_numbers = []
    times = []
    mean_angles = []
    std_angles = []
    min_angles = []
    max_angles = []
    n_angles_list = []
    si_with_bonds_list = []
    total_bonds_list = []
    all_angles_list = []  # Para guardar todos los ángulos individuales
    
    # Procesar cada frame
    if verbose:
        print(f"\nProcesando frames...")
        print("-"*70)
    
    for i, atoms in enumerate(traj):
        # Calcular tiempo
        time_ps = i * dump_interval * timestep_fs / 1000.0
        
        # Calcular ángulos
        angles, si_count, bond_count = calculate_osi_angles_along_c(
            atoms, cutoff=cutoff, c_threshold=c_threshold
        )
        
        # Guardar resultados del frame
        frame_numbers.append(i)
        times.append(time_ps)
        n_angles_list.append(len(angles))
        si_with_bonds_list.append(si_count)
        total_bonds_list.append(bond_count)
        
        if len(angles) > 0:
            mean_angles.append(np.mean(angles))
            std_angles.append(np.std(angles))
            min_angles.append(np.min(angles))
            max_angles.append(np.max(angles))
            all_angles_list.extend([(i, time_ps, angle) for angle in angles])
        else:
            mean_angles.append(np.nan)
            std_angles.append(np.nan)
            min_angles.append(np.nan)
            max_angles.append(np.nan)
        
        # Imprimir progreso
        if verbose and (i % max(1, n_frames // 20) == 0 or i == n_frames - 1):
            progress = (i + 1) / n_frames * 100
            if len(angles) > 0:
                print(f"  Frame {i+1:6d}/{n_frames} ({progress:5.1f}%) | "
                      f"t={time_ps:7.2f} ps | "
                      f"N_angles={len(angles):4d} | "
                      f"Mean={np.mean(angles):6.2f}°")
            else:
                print(f"  Frame {i+1:6d}/{n_frames} ({progress:5.1f}%) | "
                      f"t={time_ps:7.2f} ps | "
                      f"N_angles=0")
    
    elapsed_time = time.time() - start_time
    
    # Crear DataFrames
    # DataFrame 1: Estadísticas por frame
    results_per_frame = pd.DataFrame({
        'frame': frame_numbers,
        'time_ps': times,
        'mean_angle': mean_angles,
        'std_angle': std_angles,
        'min_angle': min_angles,
        'max_angle': max_angles,
        'n_angles': n_angles_list,
        'si_with_c_bonds': si_with_bonds_list,
        'total_c_bonds': total_bonds_list
    })
    
    # DataFrame 2: Todos los ángulos individuales
    all_angles_df = pd.DataFrame(all_angles_list, 
                                  columns=['frame', 'time_ps', 'angle'])
    
    if verbose:
        print("-"*70)
        print(f"\n✓ Procesamiento completado en {elapsed_time:.2f} segundos")
        print(f"\nEstadísticas globales:")
        all_angles_array = all_angles_df['angle'].values
        print(f"  Total de ángulos O-Si-O calculados: {len(all_angles_array)}")
        print(f"  Ángulo promedio: {np.mean(all_angles_array):.3f} ± {np.std(all_angles_array):.3f}°")
        print(f"  Rango: [{np.min(all_angles_array):.3f}, {np.max(all_angles_array):.3f}]°")
        print(f"  Mediana: {np.median(all_angles_array):.3f}°")
        print(f"\nÁtomos de Si con enlaces en dirección ĉ:")
        print(f"  Promedio por frame: {np.mean(si_with_bonds_list):.1f}")
        print(f"  Enlaces Si-O en ĉ por frame: {np.mean(total_bonds_list):.1f}")
    
    return results_per_frame, all_angles_df


def main():
    parser = argparse.ArgumentParser(
        description='Calcular ángulos O-Si-O en dirección ĉ desde trayectorias MD',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--traj', '-t', type=str, required=True,
                       help='Archivo de trayectoria (.traj)')
    parser.add_argument('--output', '-o', type=str, default='osi_angles_results.csv',
                       help='Archivo de salida CSV (estadísticas por frame)')
    parser.add_argument('--output-all', type=str, default='osi_angles_all.csv',
                       help='Archivo CSV con todos los ángulos individuales')
    parser.add_argument('--cutoff', type=float, default=2.0,
                       help='Distancia máxima Si-O en Å')
    parser.add_argument('--c-threshold', type=float, default=0.7,
                       help='Umbral de alineación con ĉ (cos θ). 0.7≈45°, 0.9≈25°')
    parser.add_argument('--timestep', type=float, default=0.5,
                       help='Paso de tiempo de la simulación en fs')
    parser.add_argument('--dump-interval', type=int, default=10,
                       help='Intervalo de guardado de frames')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Modo silencioso (sin output de progreso)')
    
    args = parser.parse_args()
    
    # Verificar que el archivo existe
    traj_path = Path(args.traj)
    if not traj_path.exists():
        print(f"ERROR: Archivo no encontrado: {args.traj}")
        return 1
    
    # Procesar trayectoria
    try:
        results_per_frame, all_angles_df = process_trajectory(
            args.traj,
            cutoff=args.cutoff,
            c_threshold=args.c_threshold,
            timestep_fs=args.timestep,
            dump_interval=args.dump_interval,
            verbose=not args.quiet
        )
        
        # Guardar resultados
        if not args.quiet:
            print(f"\nGuardando resultados...")
        
        results_per_frame.to_csv(args.output, index=False, float_format='%.6f')
        if not args.quiet:
            print(f"  ✓ Estadísticas por frame: {args.output}")
        
        all_angles_df.to_csv(args.output_all, index=False, float_format='%.6f')
        if not args.quiet:
            print(f"  ✓ Todos los ángulos: {args.output_all}")
        
        if not args.quiet:
            print("\n" + "="*70)
            print(" ✅ CÁLCULO COMPLETADO")
            print("="*70)
        
        return 0
        
    except Exception as e:
        print(f"\nERROR durante el procesamiento: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
