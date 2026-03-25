#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para verificar la relación O:Si = 2:1 en archivos XML de VASP.
Identifica archivos que no cumplen la relación esperada.
"""

import os
import sys
import argparse
from ase.io import read
from collections import Counter


def analyze_xml_si_o_ratio(xml_file):
    """
    Analiza la relación O:Si en un archivo XML de VASP.
    
    Args:
        xml_file: Ruta al archivo XML
        
    Returns:
        dict: Información sobre el análisis de la relación
    """
    try:
        # Leer solo la primera estructura para análisis de composición
        atoms = read(xml_file, index=0)
        symbols = atoms.get_chemical_symbols()
        element_counts = Counter(symbols)
        
        si_count = element_counts.get('Si', 0)
        o_count = element_counts.get('O', 0)
        
        result = {
            'file': xml_file,
            'si_count': si_count,
            'o_count': o_count,
            'total_atoms': len(atoms),
            'has_si': si_count > 0,
            'has_o': o_count > 0,
            'ratio_ok': False,
            'actual_ratio': 0.0,
            'expected_ratio': 2.0,
            'other_elements': {k: v for k, v in element_counts.items() if k not in ['Si', 'O']},
            'error': None
        }
        
        if si_count > 0:
            result['actual_ratio'] = o_count / si_count
            result['ratio_ok'] = abs(result['actual_ratio'] - 2.0) < 1e-6
        
        return result
        
    except Exception as e:
        return {
            'file': xml_file,
            'error': str(e),
            'ratio_ok': False
        }


def find_xml_files(data_path):
    """Buscar archivos XML de VASP en el directorio."""
    xml_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith("run.xml") or file.endswith(".xml"):
                xml_files.append(os.path.join(root, file))
    return xml_files


def check_xml_si_o_ratios(data_path, tolerance=0.0, verbose=False, show_only_problems=True):
    """
    Verifica la relación O:Si en todos los archivos XML de un directorio.
    
    Args:
        data_path (str): Directorio que contiene archivos XML
        tolerance (float): Tolerancia para considerar la relación como correcta
        verbose (bool): Mostrar información detallada
        show_only_problems (bool): Solo mostrar archivos problemáticos
        
    Returns:
        dict: Estadísticas del análisis
    """
    print(f"🔍 Buscando archivos XML en: {data_path}")
    
    xml_files = find_xml_files(data_path)
    
    if not xml_files:
        print(f"❌ No se encontraron archivos XML en {data_path}")
        return None
    
    print(f"📁 Encontrados {len(xml_files)} archivos XML")
    print("=" * 80)
    
    correct_ratio_files = 0
    problematic_files = []
    error_files = []
    
    for xml_file in xml_files:
        if verbose:
            print(f"📂 Analizando: {xml_file}")
        
        analysis = analyze_xml_si_o_ratio(xml_file)
        
        if analysis.get('error'):
            error_files.append(analysis)
            if show_only_problems or verbose:
                print(f"💥 ERROR - {xml_file}")
                print(f"   Razón: {analysis['error']}")
            continue
        
        if analysis['has_si'] and analysis['has_o']:
            if abs(analysis['actual_ratio'] - 2.0) <= tolerance:
                correct_ratio_files += 1
                if not show_only_problems or verbose:
                    print(f"✅ OK - {xml_file}")
                    print(f"   Si: {analysis['si_count']}, O: {analysis['o_count']}, Ratio: {analysis['actual_ratio']:.3f}")
            else:
                problematic_files.append(analysis)
                if show_only_problems or verbose:
                    print(f"❌ PROBLEMA - {xml_file}")
                    print(f"   Si: {analysis['si_count']}, O: {analysis['o_count']}, Ratio: {analysis['actual_ratio']:.3f} (esperado: 2.0)")
                    if analysis['other_elements']:
                        other = ", ".join([f"{k}={v}" for k, v in analysis['other_elements'].items()])
                        print(f"   Otros elementos: {other}")
        else:
            problematic_files.append(analysis)
            if show_only_problems or verbose:
                print(f"❌ PROBLEMA - {xml_file}")
                if not analysis['has_si'] and not analysis['has_o']:
                    print(f"   Sin Si ni O en la estructura")
                elif not analysis['has_si']:
                    print(f"   Sin Si: solo O={analysis['o_count']}")
                else:
                    print(f"   Sin O: solo Si={analysis['si_count']}")
    
    # Mostrar resumen
    total_files = len(xml_files)
    print("\n" + "=" * 80)
    print("📊 RESUMEN DE ANÁLISIS DE RELACIÓN O:Si")
    print("=" * 80)
    print(f"Total de archivos XML analizados: {total_files}")
    print(f"Archivos con relación O:Si = 2.0 (±{tolerance}): {correct_ratio_files}")
    print(f"Archivos problemáticos: {len(problematic_files)}")
    print(f"Archivos con errores de lectura: {len(error_files)}")
    
    if correct_ratio_files > 0:
        percentage = (correct_ratio_files / total_files) * 100
        print(f"✅ Porcentaje con relación correcta: {percentage:.1f}%")
    
    if problematic_files and show_only_problems:
        print(f"\n❌ ARCHIVOS CON PROBLEMAS DE RELACIÓN O:Si:")
        print("-" * 80)
        for analysis in problematic_files:
            print(f"📁 {analysis['file']}")
            if analysis.get('has_si') and analysis.get('has_o'):
                print(f"   Si: {analysis['si_count']}, O: {analysis['o_count']}, Ratio: {analysis['actual_ratio']:.3f}")
            else:
                print(f"   Si: {analysis.get('si_count', 0)}, O: {analysis.get('o_count', 0)} (composición incompleta)")
    
    if error_files:
        print(f"\n💥 ARCHIVOS CON ERRORES DE LECTURA:")
        print("-" * 80)
        for analysis in error_files:
            print(f"📁 {analysis['file']}")
            print(f"   Error: {analysis['error']}")
    
    print("=" * 80)
    
    return {
        'total_files': total_files,
        'correct_ratio_files': correct_ratio_files,
        'problematic_files': problematic_files,
        'error_files': error_files,
        'success_rate': (correct_ratio_files / total_files) * 100 if total_files > 0 else 0
    }


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Verificar relación O:Si = 2:1 en archivos XML de VASP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Buscar archivos problemáticos en directorio actual
  %(prog)s .
  
  # Analizar directorio específico con tolerancia
  %(prog)s /path/to/vasp/files --tolerance 0.1
  
  # Mostrar todos los archivos (no solo problemáticos)
  %(prog)s /path/to/vasp/files --show-all --verbose
  
  # Solo archivos con problemas (modo por defecto)
  %(prog)s /path/to/vasp/files --problems-only
        """
    )
    
    parser.add_argument(
        'data_path',
        help='Directorio que contiene archivos XML de VASP'
    )
    
    parser.add_argument(
        '--tolerance', '-t',
        type=float,
        default=0.0,
        help='Tolerancia para la relación O:Si (default: 0.0)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mostrar información detallada de cada archivo'
    )
    
    parser.add_argument(
        '--show-all',
        action='store_true',
        help='Mostrar todos los archivos (no solo problemáticos)'
    )
    
    parser.add_argument(
        '--problems-only',
        action='store_true',
        help='Solo mostrar archivos problemáticos (default)'
    )
    
    args = parser.parse_args()
    
    # Verificar que el directorio existe
    if not os.path.isdir(args.data_path):
        print(f"❌ Error: Directorio '{args.data_path}' no encontrado")
        return
    
    # Determinar modo de visualización
    show_only_problems = not args.show_all
    if args.problems_only:
        show_only_problems = True
    
    # Ejecutar análisis
    results = check_xml_si_o_ratios(
        args.data_path, 
        tolerance=args.tolerance,
        verbose=args.verbose,
        show_only_problems=show_only_problems
    )
    
    if results:
        print(f"\n📋 RESULTADO FINAL:")
        if results['success_rate'] == 100.0:
            print("🎉 ¡Perfecto! Todos los archivos XML tienen la relación O:Si = 2:1")
        elif results['success_rate'] > 80.0:
            print(f"👍 Bien: {results['success_rate']:.1f}% de los archivos tienen la relación correcta")
        else:
            print(f"⚠️  Atención: Solo {results['success_rate']:.1f}% de los archivos tienen la relación correcta")
        
        if results['problematic_files']:
            print(f"\n🗑️  Archivos problemáticos encontrados: {len(results['problematic_files'])}")
            print("   Estos archivos podrían necesitar revisión o exclusión del dataset")


if __name__ == "__main__":
    main()