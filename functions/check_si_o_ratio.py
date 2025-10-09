#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para filtrar estructuras atómicas manteniendo solo aquellas con relación O:Si = 2:1.
Por defecto sobreescribe el archivo original con solo las configuracion    print("EJEMPLOS:")
    print("  python check_si_o_ratio.py silica.xyz                    # Sobreescribir por defecto")
    print("  python check_si_o_ratio.py quartz.xyz --verbose          # Con detalles")
    print("  python check_si_o_ratio.py glass.xyz --tolerance 0.1     # Con tolerancia")
    print("  python check_si_o_ratio.py data.xyz --no-backup          # Sin backup descartadas")
    print("  python check_si_o_ratio.py data.xyz --analysis-only      # Solo análisis")rectas.
"""

import sys
import os
from ase.io import read, write
from tqdm import tqdm
from collections import Counter


def analyze_si_o_ratio(atoms):
    """
    Analiza la relación O:Si en una estructura atómica.
    
    Args:
        atoms: Objeto Atoms de ASE
        
    Returns:
        dict: Información sobre el análisis de la relación
    """
    symbols = atoms.get_chemical_symbols()
    element_counts = Counter(symbols)
    
    si_count = element_counts.get('Si', 0)
    o_count = element_counts.get('O', 0)
    
    result = {
        'si_count': si_count,
        'o_count': o_count,
        'total_atoms': len(atoms),
        'has_si': si_count > 0,
        'has_o': o_count > 0,
        'ratio_ok': False,
        'actual_ratio': 0.0,
        'expected_ratio': 2.0,
        'other_elements': {k: v for k, v in element_counts.items() if k not in ['Si', 'O']}
    }
    
    if si_count > 0:
        result['actual_ratio'] = o_count / si_count
        result['ratio_ok'] = abs(result['actual_ratio'] - 2.0) < 1e-6
    
    return result


def check_file_si_o_ratio(input_file, verbose=False, tolerance=0.0, filter_mode=True, backup_discarded=True):
    """
    Verifica y filtra la relación O:Si en todas las estructuras de un archivo.
    Por defecto sobreescribe el archivo original con solo las configuraciones O:Si = 2:1.
    
    Args:
        input_file (str): Archivo de entrada (será sobreescrito)
        verbose (bool): Mostrar detalles de cada frame
        tolerance (float): Tolerancia para considerar la relación como correcta
        filter_mode (bool): Si True, sobreescribe con configuraciones correctas (default: True)
        backup_discarded (bool): Si True, guarda configuraciones descartadas en archivo separado
        
    Returns:
        dict: Estadísticas del análisis
    """
    if filter_mode:
        print(f"🔧 Filtrando configuraciones con relación O:Si = 2:1 en {input_file}...")
    else:
        print(f"📊 Analizando relación O:Si en {input_file}...")
    
    try:
        #data = read(input_file, index=':', format='extxyz')
        data = read(input_file, index=':')
    except Exception as e:
        print(f"❌ Error leyendo archivo: {e}")
        return None
    
    total_frames = len(data)
    correct_ratio_frames = 0
    frames_with_si = 0
    frames_with_o = 0
    problematic_frames = []
    correct_configurations = []  # Para guardar configuraciones correctas
    incorrect_configurations = []  # Para guardar configuraciones incorrectas
    
    if filter_mode:
        print(f"🔍 Procesando {total_frames} frames para filtrado...")
    else:
        print(f"🔍 Analizando {total_frames} frames...")
    
    for i, atoms in enumerate(tqdm(data, desc="Verificando O:Si", unit="frame")):
        analysis = analyze_si_o_ratio(atoms)
        
        if analysis['has_si']:
            frames_with_si += 1
            
        if analysis['has_o']:
            frames_with_o += 1
            
        if analysis['has_si'] and analysis['has_o']:
            if abs(analysis['actual_ratio'] - 2.0) <= tolerance:
                correct_ratio_frames += 1
                if filter_mode:
                    correct_configurations.append(atoms)
            else:
                problematic_frames.append({
                    'frame': i + 1,
                    'si_count': analysis['si_count'],
                    'o_count': analysis['o_count'],
                    'actual_ratio': analysis['actual_ratio'],
                    'other_elements': analysis['other_elements']
                })
                if filter_mode:
                    incorrect_configurations.append(atoms)
        elif analysis['has_si'] or analysis['has_o']:
            # Frame tiene Si o O pero no ambos
            problematic_frames.append({
                'frame': i + 1,
                'si_count': analysis['si_count'],
                'o_count': analysis['o_count'],
                'actual_ratio': analysis['actual_ratio'],
                'other_elements': analysis['other_elements']
            })
            if filter_mode:
                incorrect_configurations.append(atoms)
        else:
            # Frame sin Si ni O
            if filter_mode:
                incorrect_configurations.append(atoms)
        
        if verbose and (analysis['has_si'] or analysis['has_o']):
            status = "✅" if analysis['ratio_ok'] else "❌"
            print(f"  {status} Frame {i+1}: Si={analysis['si_count']}, O={analysis['o_count']}, "
                  f"Ratio={analysis['actual_ratio']:.2f}")
    
    # Mostrar estadísticas
    print(f"\n{'='*60}")
    if filter_mode:
        print("� ESTADÍSTICAS DE FILTRADO O:Si")
    else:
        print("📈 ESTADÍSTICAS DE ANÁLISIS O:Si")
    print("="*60)
    print(f"Total de frames analizados: {total_frames}")
    print(f"Frames con Si: {frames_with_si}")
    print(f"Frames con O: {frames_with_o}")
    print(f"Frames con relación O:Si = 2.0 (±{tolerance}): {correct_ratio_frames}")
    print(f"Frames problemáticos: {len(problematic_frames)}")
    
    if correct_ratio_frames > 0:
        percentage = (correct_ratio_frames / total_frames) * 100
        print(f"✅ Porcentaje con relación correcta: {percentage:.1f}%")
    
    if problematic_frames:
        print(f"\n❌ FRAMES PROBLEMÁTICOS ({len(problematic_frames)}):")
        print("-" * 60)
        for frame_info in problematic_frames[:10]:  # Mostrar solo los primeros 10
            print(f"Frame {frame_info['frame']}: Si={frame_info['si_count']}, "
                  f"O={frame_info['o_count']}, Ratio={frame_info['actual_ratio']:.3f}")
            if frame_info['other_elements']:
                other = ", ".join([f"{k}={v}" for k, v in frame_info['other_elements'].items()])
                print(f"  Otros elementos: {other}")
        
        if len(problematic_frames) > 10:
            print(f"  ... y {len(problematic_frames) - 10} más")
    
    # Guardar archivos filtrados si está en modo filtro
    if filter_mode:
        if correct_configurations:
            print(f"\n💾 Sobreescribiendo {input_file} con {len(correct_configurations)} configuraciones limpias...")
            write(input_file, correct_configurations)
            print(f"✅ Archivo sobreescrito: {input_file}")
        else:
            print("\n❌ No se encontraron configuraciones con relación O:Si correcta")
            print(f"⚠️  El archivo {input_file} quedaría vacío. No se modifica.")
            
        # Guardar configuraciones descartadas como backup si se especifica
        if incorrect_configurations and backup_discarded:
            base_name, ext = os.path.splitext(input_file)
            discarded_file = f"{base_name}_discarded{ext}"
            print(f"\n🗑️  Guardando {len(incorrect_configurations)} configuraciones descartadas en {discarded_file}...")
            write(discarded_file, incorrect_configurations)
            print(f"📁 Backup de descartadas guardado: {discarded_file}")
    
    print("="*60)
    
    return {
        'total_frames': total_frames,
        'frames_with_si': frames_with_si,
        'frames_with_o': frames_with_o,
        'correct_ratio_frames': correct_ratio_frames,
        'problematic_frames': problematic_frames,
        'success_rate': (correct_ratio_frames / total_frames) * 100 if total_frames > 0 else 0,
        'overwritten': filter_mode and correct_configurations,
        'discarded_file': discarded_file if filter_mode and incorrect_configurations and backup_discarded else None
    }


def show_help():
    """Mostrar mensaje de ayuda."""
    print("="*60)
    print("        LIMPIADOR DE RELACIÓN O:Si = 2:1")
    print("="*60)
    print("DESCRIPCIÓN:")
    print("  Filtra configuraciones atómicas manteniendo solo aquellas con")
    print("  relación O:Si = 2:1. Por defecto SOBREESCRIBE el archivo original.")
    print()
    print("USO:")
    print("  python check_si_o_ratio.py <archivo.xyz> [opciones]")
    print()
    print("OPCIONES:")
    print("  -v, --verbose       Mostrar detalles de cada frame")
    print("  -t, --tolerance     Tolerancia para la relación (default: 0.0)")
    print("  --no-backup         No guardar configuraciones descartadas")
    print("  --analysis-only     Solo analizar sin modificar archivo")
    print("  -h, --help          Mostrar esta ayuda")
    print()
    print("EJEMPLOS:")
    print("  python check_si_o_ratio.py silica.xyz                    # Filtrar por defecto")
    print("  python check_si_o_ratio.py quartz.xyz --verbose          # Con detalles")
    print("  python check_si_o_ratio.py glass.xyz --tolerance 0.1     # Con tolerancia")
    print("  python check_si_o_ratio.py data.xyz --output clean.xyz   # Salida personalizada")
    print("  python check_si_o_ratio.py data.xyz --analysis-only      # Solo análisis")
    print()
    print("NOTAS:")
    print("  • ⚠️  Por defecto SOBREESCRIBE el archivo original")
    print("  • Relación esperada: O:Si = 2:1 (típica de SiO₂)")
    print("  • Crea backup de descartadas: *_discarded.xyz (usar --no-backup para evitar)")
    print("  • Usa --analysis-only para solo analizar sin modificar archivos")
    print("="*60)


def main():
    """Función principal."""
    if len(sys.argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
        show_help()
        return
    
    input_file = sys.argv[1]
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    filter_mode = not ('--analysis-only' in sys.argv)  # Por defecto True, False solo si --analysis-only
    backup_discarded = False if not ('--no-backup' in sys.argv) else True  # Por defecto False, True solo si --no-backup
    tolerance = 0.0
    
    # Buscar tolerancia
    for i, arg in enumerate(sys.argv):
        if arg in ['--tolerance', '-t'] and i + 1 < len(sys.argv):
            try:
                tolerance = float(sys.argv[i + 1])
            except ValueError:
                print("❌ Error: Tolerancia debe ser un número")
                return
    
    if not os.path.isfile(input_file):
        print(f"❌ Error: Archivo '{input_file}' no encontrado")
        return
        
    # Sin advertencias
    
    # Ejecutar análisis
    results = check_file_si_o_ratio(input_file, verbose=verbose, tolerance=tolerance, 
                                  filter_mode=filter_mode, backup_discarded=backup_discarded)
    
    if results:
        if filter_mode:
            if results['success_rate'] == 100.0:
                print("\n🎉 ¡Perfecto! Todas las estructuras ya tenían la relación O:Si = 2:1")
                print(f"📄 Archivo {input_file} no fue modificado (ya estaba limpio)")
            elif results.get('overwritten'):
                print(f"\n✅ Filtrado completado: {results['success_rate']:.1f}% de las estructuras eran correctas")
                print(f"� Archivo {input_file} sobreescrito con configuraciones limpias")
            else:
                print(f"\n❌ No se encontraron configuraciones correctas")
                print(f"� Archivo {input_file} NO fue modificado")
                
            if results.get('discarded_file'):
                print(f"🗑️  Configuraciones descartadas respaldadas en: {results['discarded_file']}")
        else:
            # Solo análisis
            if results['success_rate'] == 100.0:
                print("\n🎉 ¡Perfecto! Todas las estructuras tienen la relación O:Si = 2:1")
            elif results['success_rate'] > 80.0:
                print(f"\n👍 Bien: {results['success_rate']:.1f}% de las estructuras tienen la relación correcta")
            else:
                print(f"\n⚠️  Atención: Solo {results['success_rate']:.1f}% de las estructuras tienen la relación correcta")
            print("\n💡 Ejecuta sin --analysis-only para limpiar el archivo automáticamente")


if __name__ == "__main__":
    main()
