#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XYZ Comparator

A command-line tool to compare reference and predicted properties from XYZ files.
Supports both combined files (with REF_ and PRED_ prefixes) and separate files.

Usage:
    python xyz_comparator.py --combined test-evaluated.xyz --ref-prefix REF --pred-prefix MACE
    python xyz_comparator.py --ref ref.xyz --pred mace.xyz --properties energy forces
    python xyz_comparator.py --help
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from ase import Atoms
from typing import List, Dict, Optional
import os
import sys
from sklearn.metrics import r2_score


class XYZComparator:
    def __init__(self):
        self.colors = [
            "#1f77b4",  # muted blue (reference)
            "#d62728",  # brick red (predictions)
            "#2ca02c",  # green
        ]
        
        # Set matplotlib parameters
        plt.rcParams.update({"font.size": 12})
    
    def load_xyz_properties(self, filepath: str, prefix: str = "REF") -> Dict[str, np.ndarray]:
        """
        Load properties from XYZ file.
        
        Args:
            filepath: Path to XYZ file
            prefix: Property prefix ("REF", "MACE", etc.)
        
        Returns:
            Dictionary with:
            - 'energy': array of energies
            - 'energy_per_atom': array of energies per atom
            - 'forces': flattened array of all force components
            - 'stress': flattened array of all stress components (if available)
            - 'num_atoms': array with number of atoms per configuration
        """
        try:
            atoms_list = read(filepath, index=':', format='extxyz')
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            sys.exit(1)
        
        results = {
            'energy': [],
            'energy_per_atom': [],
            'forces': [],
            'stress': [],
            'num_atoms': []
        }
        
        for atoms in atoms_list:
            # Get number of atoms
            natoms = len(atoms)
            results['num_atoms'].append(natoms)
            
            # Get energy
            energy_key = f'{prefix}_energy'
            if energy_key in atoms.info:
                energy = atoms.info[energy_key]
                results['energy'].append(energy)
                results['energy_per_atom'].append(energy / natoms)
            
            # Get forces
            forces_key = f'{prefix}_forces'
            if hasattr(atoms, 'arrays') and forces_key in atoms.arrays:
                forces = atoms.arrays[forces_key]
                results['forces'].extend(forces.flatten())
            
            # Get stress
            stress_key = f'{prefix}_stress'
            if stress_key in atoms.info:
                stress_raw = atoms.info[stress_key]
                
                # Handle different stress formats
                if isinstance(stress_raw, str):
                    # Handle JSON format (e.g., "_JSON [[...]]")
                    if stress_raw.startswith('_JSON'):
                        import json
                        stress_data = json.loads(stress_raw[6:])  # Remove "_JSON " prefix
                        stress = np.array(stress_data).flatten()
                    else:
                        # Handle space-separated string format
                        stress = np.fromstring(stress_raw, sep=' ')
                else:
                    # Handle direct array format
                    stress = np.array(stress_raw).flatten()
                
                # Convert from GPa to eV/Å³ if needed (1 GPa = 0.0062415091 eV/Å³)
                if np.max(np.abs(stress)) > 1:  # Stress values > 1 are likely in GPa
                    stress = stress * 0.0062415091
                    
                results['stress'].extend(stress)
        
        # Convert lists to numpy arrays
        for key in results:
            if results[key]:  # Only convert if list is not empty
                results[key] = np.array(results[key])
            else:
                results[key] = None
                
        return results
    
    def _calculate_r2(self, y_true, y_pred):
        """Calculate R² (coefficient of determination) using sklearn."""
        return r2_score(y_true, y_pred)
    
    def compare_xyz_files(
        self,
        ref_xyz: Optional[str] = None,
        pred_xyz: Optional[str] = None,
        combined_xyz: Optional[str] = None,
        ref_prefix: str = "REF",
        pred_prefix: str = "MACE",
        properties: List[str] = ["energy", "forces", "stress"],
        plot_path: Optional[str] = None,
        show_plots: bool = True
    ):
        """
        Compare reference and predicted XYZ files.
        
        Args:
            ref_xyz: Path to reference XYZ file (if using separate files)
            pred_xyz: Path to predicted XYZ file (if using separate files)
            combined_xyz: Path to XYZ file containing both ref and pred data
            ref_prefix: Prefix for reference properties (e.g., "REF")
            pred_prefix: Prefix for predicted properties (e.g., "MACE")
            properties: List of properties to compare
            plot_path: If provided, save plots to this path
            show_plots: Whether to display the plots
        """
        # Load data
        if combined_xyz is not None:
            # Load both reference and predicted data from the same file
            print(f"Loading data from combined file: {combined_xyz}")
            ref_data = self.load_xyz_properties(combined_xyz, prefix=ref_prefix)
            pred_data = self.load_xyz_properties(combined_xyz, prefix=pred_prefix)
        elif ref_xyz is not None and pred_xyz is not None:
            # Load data from separate files
            print(f"Loading reference data from: {ref_xyz}")
            print(f"Loading predicted data from: {pred_xyz}")
            ref_data = self.load_xyz_properties(ref_xyz, prefix=ref_prefix)
            pred_data = self.load_xyz_properties(pred_xyz, prefix=pred_prefix)
        else:
            raise ValueError("Either provide combined_xyz or both ref_xyz and pred_xyz")
        
        # Synchronize datasets to the same size
        ref_data, pred_data = self.synchronize_datasets(ref_data, pred_data, verbose=True)
        
        # Create subplots
        n_props = len(properties)
        fig, axes = plt.subplots(1, n_props, figsize=(6*n_props, 6))
        if n_props == 1:
            axes = [axes]
        
        # Plot each property
        for ax, prop in zip(axes, properties):
            if prop == "energy":
                self._plot_energy_comparison(ax, ref_data, pred_data)
            elif prop == "forces":
                self._plot_forces_comparison(ax, ref_data, pred_data)
            elif prop == "stress":
                self._plot_stress_comparison(ax, ref_data, pred_data)
        
        plt.tight_layout()
        
        # Save or show plots
        if plot_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(plot_path) if os.path.dirname(plot_path) else '.', exist_ok=True)
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            print(f"Plots saved to {plot_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Calculate and print some metrics
        self._print_comparison_metrics(ref_data, pred_data)
    
    def _plot_energy_comparison(self, ax, ref_data, pred_data):
        """Plot energy per atom comparison."""
        if ref_data['energy_per_atom'] is None or pred_data['energy_per_atom'] is None:
            ax.text(0.5, 0.5, "No energy data", ha='center', va='center')
            return
        
        # Scatter plot
        ax.scatter(
            ref_data['energy_per_atom'],
            pred_data['energy_per_atom'],
            color=self.colors[0],
            alpha=0.6,
            label='Energy per atom'
        )
        
        # Add diagonal and labels
        self._add_diagonal(ax)
        ax.set_xlabel("Reference Energy per atom [eV]")
        ax.set_ylabel("Predicted Energy per atom [eV]")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Add RMSE and R² to plot
        rmse = np.sqrt(np.mean((ref_data['energy_per_atom'] - pred_data['energy_per_atom'])**2)) * 1000
        r2 = self._calculate_r2(ref_data['energy_per_atom'], pred_data['energy_per_atom'])
        
        text_str = f"RMSE: {rmse:.2f} meV/atom\nR²: {r2:.4f}"
        ax.text(0.05, 0.95, text_str, 
                transform=ax.transAxes, ha='left', va='top',
                bbox=dict(facecolor='white', alpha=0.8))
    
    def _plot_forces_comparison(self, ax, ref_data, pred_data):
        """Plot forces comparison."""
        if ref_data['forces'] is None or pred_data['forces'] is None:
            ax.text(0.5, 0.5, "No forces data", ha='center', va='center')
            return
        
        # Scatter plot (use alpha for large datasets)
        alpha = 0.3 if len(ref_data['forces']) > 1000 else 0.6
        ax.scatter(
            ref_data['forces'],
            pred_data['forces'],
            color=self.colors[1],
            alpha=alpha,
            label='Forces'
        )
        
        # Add diagonal and labels
        self._add_diagonal(ax)
        ax.set_xlabel("Reference Forces [eV/Å]")
        ax.set_ylabel("Predicted Forces [eV/Å]")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Add RMSE and R² to plot
        rmse = np.sqrt(np.mean((ref_data['forces'] - pred_data['forces'])**2))
        r2 = self._calculate_r2(ref_data['forces'], pred_data['forces'])
        
        text_str = f"RMSE: {rmse:.3f} eV/Å\nR²: {r2:.4f}"
        ax.text(0.05, 0.95, text_str, 
                transform=ax.transAxes, ha='left', va='top',
                bbox=dict(facecolor='white', alpha=0.8))
    
    def _plot_stress_comparison(self, ax, ref_data, pred_data):
        """Plot stress comparison."""
        if ref_data['stress'] is None or pred_data['stress'] is None:
            ax.text(0.5, 0.5, "No stress data", ha='center', va='center')
            return
        
        # Scatter plot
        ax.scatter(
            ref_data['stress'],
            pred_data['stress'],
            color=self.colors[2],
            alpha=0.6,
            label='Stress'
        )
        
        # Add diagonal and labels
        self._add_diagonal(ax)
        ax.set_xlabel("Reference Stress [eV/Å³]")
        ax.set_ylabel("Predicted Stress [eV/Å³]")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Add RMSE and R² to plot
        rmse = np.sqrt(np.mean((ref_data['stress'] - pred_data['stress'])**2)) * 1000
        r2 = self._calculate_r2(ref_data['stress'], pred_data['stress'])
        
        text_str = f"RMSE: {rmse:.2f} meV/Å³\nR²: {r2:.4f}"
        ax.text(0.05, 0.95, text_str, 
                transform=ax.transAxes, ha='left', va='top',
                bbox=dict(facecolor='white', alpha=0.8))
    
    def _add_diagonal(self, ax):
        """Add diagonal line to a plot."""
        min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--",
            color="black",
            alpha=0.7,
        )
    
    def _print_comparison_metrics(self, ref_data, pred_data):
        """Print some comparison metrics."""
        print("\nComparison Metrics:")
        print("------------------")
        
        # Energy metrics
        if ref_data['energy_per_atom'] is not None and pred_data['energy_per_atom'] is not None:
            diff = ref_data['energy_per_atom'] - pred_data['energy_per_atom']
            rmse = np.sqrt(np.mean(diff**2)) * 1000  # in meV/atom
            mae = np.mean(np.abs(diff)) * 1000  # in meV/atom
            r2 = self._calculate_r2(ref_data['energy_per_atom'], pred_data['energy_per_atom'])
            print(f"Energy per atom:")
            print(f"  RMSE: {rmse:.2f} meV/atom")
            print(f"  MAE:  {mae:.2f} meV/atom")
            print(f"  R²:   {r2:.4f}")
            print(f"  Max error: {np.max(np.abs(diff))*1000:.2f} meV/atom")
        
        # Forces metrics
        if ref_data['forces'] is not None and pred_data['forces'] is not None:
            diff = ref_data['forces'] - pred_data['forces']
            rmse = np.sqrt(np.mean(diff**2))
            mae = np.mean(np.abs(diff))
            r2 = self._calculate_r2(ref_data['forces'], pred_data['forces'])
            print(f"\nForces:")
            print(f"  RMSE: {rmse:.4f} eV/Å")
            print(f"  MAE:  {mae:.4f} eV/Å")
            print(f"  R²:   {r2:.4f}")
            print(f"  Max error: {np.max(np.abs(diff)):.4f} eV/Å")
        
        # Stress metrics
        if ref_data['stress'] is not None and pred_data['stress'] is not None:
            diff = ref_data['stress'] - pred_data['stress']
            rmse = np.sqrt(np.mean(diff**2)) * 1000  # in meV/Å³
            mae = np.mean(np.abs(diff)) * 1000  # in meV/Å³
            r2 = self._calculate_r2(ref_data['stress'], pred_data['stress'])
            print(f"\nStress:")
            print(f"  RMSE: {rmse:.2f} meV/Å³")
            print(f"  MAE:  {mae:.2f} meV/Å³")
            print(f"  R²:   {r2:.4f}")
            print(f"  Max error: {np.max(np.abs(diff))*1000:.2f} meV/Å³")
    
    def analyze_xyz_properties(self, filepath: str) -> Dict[str, List[str]]:
        """
        Analyze what properties are available in an XYZ file.
        
        Args:
            filepath: Path to XYZ file
            
        Returns:
            Dictionary with 'info_keys' and 'array_keys' available in the file
        """
        try:
            atoms_list = read(filepath, index='0', format='extxyz')  # Read just first frame
            atoms = atoms_list if isinstance(atoms_list, Atoms) else atoms_list[0]
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            sys.exit(1)
        
        info_keys = list(atoms.info.keys())
        array_keys = list(atoms.arrays.keys()) if hasattr(atoms, 'arrays') else []
        
        print(f"Analysis of {filepath}:")
        print(f"  Info keys: {info_keys}")
        print(f"  Array keys: {array_keys}")
        print(f"  Number of atoms: {len(atoms)}")
        
        return {'info_keys': info_keys, 'array_keys': array_keys}
    
    def debug_compare_data_loading(
        self,
        ref_xyz: Optional[str] = None,
        pred_xyz: Optional[str] = None,
        combined_xyz: Optional[str] = None,
        ref_prefix: str = "REF",
        pred_prefix: str = "MACE"
    ):
        """
        Debug function to compare data loading between combined and separate files.
        """
        print("=== DEBUG: Comparing data loading methods ===")
        
        # Method 1: Combined file
        if combined_xyz:
            print(f"\n1. Loading from combined file: {combined_xyz}")
            ref_data_combined = self.load_xyz_properties(combined_xyz, prefix=ref_prefix)
            pred_data_combined = self.load_xyz_properties(combined_xyz, prefix=pred_prefix)
            
            print(f"   REF data shapes:")
            for key, data in ref_data_combined.items():
                if data is not None:
                    print(f"     {key}: {data.shape if hasattr(data, 'shape') else len(data)}")
                else:
                    print(f"     {key}: None")
            
            print(f"   PRED data shapes:")
            for key, data in pred_data_combined.items():
                if data is not None:
                    print(f"     {key}: {data.shape if hasattr(data, 'shape') else len(data)}")
                else:
                    print(f"     {key}: None")
        
        # Method 2: Separate files
        if ref_xyz and pred_xyz:
            print(f"\n2. Loading from separate files: {ref_xyz} and {pred_xyz}")
            ref_data_separate = self.load_xyz_properties(ref_xyz, prefix=ref_prefix)
            pred_data_separate = self.load_xyz_properties(pred_xyz, prefix=pred_prefix)
            
            print(f"   REF data shapes:")
            for key, data in ref_data_separate.items():
                if data is not None:
                    print(f"     {key}: {data.shape if hasattr(data, 'shape') else len(data)}")
                else:
                    print(f"     {key}: None")
            
            print(f"   PRED data shapes:")
            for key, data in pred_data_separate.items():
                if data is not None:
                    print(f"     {key}: {data.shape if hasattr(data, 'shape') else len(data)}")
                else:
                    print(f"     {key}: None")
        
        # Compare if both methods available
        if combined_xyz and ref_xyz and pred_xyz:
            print(f"\n3. Comparing first few values:")
            
            # Compare energies
            if ref_data_combined['energy'] is not None and ref_data_separate['energy'] is not None:
                print(f"   REF energies match: {np.allclose(ref_data_combined['energy'][:5], ref_data_separate['energy'][:5])}")
                print(f"     Combined first 3: {ref_data_combined['energy'][:3]}")
                print(f"     Separate first 3: {ref_data_separate['energy'][:3]}")
            
            if pred_data_combined['energy'] is not None and pred_data_separate['energy'] is not None:
                print(f"   PRED energies match: {np.allclose(pred_data_combined['energy'][:5], pred_data_separate['energy'][:5])}")
                print(f"     Combined first 3: {pred_data_combined['energy'][:3]}")
                print(f"     Separate first 3: {pred_data_separate['energy'][:3]}")
            
            # Compare forces
            if ref_data_combined['forces'] is not None and ref_data_separate['forces'] is not None:
                forces_match = np.allclose(ref_data_combined['forces'][:10], ref_data_separate['forces'][:10])
                print(f"   REF forces match: {forces_match}")
                if not forces_match:
                    print(f"     Combined first 3: {ref_data_combined['forces'][:3]}")
                    print(f"     Separate first 3: {ref_data_separate['forces'][:3]}")
            
            if pred_data_combined['forces'] is not None and pred_data_separate['forces'] is not None:
                forces_match = np.allclose(pred_data_combined['forces'][:10], pred_data_separate['forces'][:10])
                print(f"   PRED forces match: {forces_match}")
                if not forces_match:
                    print(f"     Combined first 3: {pred_data_combined['forces'][:3]}")
                    print(f"     Separate first 3: {pred_data_separate['forces'][:3]}")


    def synchronize_datasets(self, ref_data, pred_data, verbose=False):
        """
        Ensure both datasets have the same number of data points.
        Truncates to the minimum size if they differ.
        """
        if verbose:
            print(f"\n=== Dataset Synchronization ===")
        
        # Check sizes for each property
        properties_to_sync = ['energy', 'energy_per_atom', 'num_atoms']
        min_size = None
        
        for prop in properties_to_sync:
            if ref_data[prop] is not None and pred_data[prop] is not None:
                ref_size = len(ref_data[prop])
                pred_size = len(pred_data[prop])
                current_min = min(ref_size, pred_size)
                
                if min_size is None:
                    min_size = current_min
                else:
                    min_size = min(min_size, current_min)
                
                if verbose:
                    print(f"  {prop}: REF={ref_size}, PRED={pred_size}, min={current_min}")
        
        if min_size is None:
            if verbose:
                print("  No compatible properties found")
            return ref_data, pred_data
        
        if verbose:
            print(f"  Synchronizing all arrays to size: {min_size}")
        
        # Synchronize all properties
        ref_sync = {}
        pred_sync = {}
        
        for prop in ref_data.keys():
            if ref_data[prop] is not None and pred_data[prop] is not None:
                if prop in ['forces', 'stress']:
                    # For forces and stress, calculate how many elements per structure
                    if prop == 'forces':
                        elements_per_structure = len(ref_data[prop]) // len(ref_data['num_atoms'])
                        ref_sync[prop] = ref_data[prop][:min_size * elements_per_structure]
                        pred_sync[prop] = pred_data[prop][:min_size * elements_per_structure]
                    elif prop == 'stress':
                        stress_per_structure = len(ref_data[prop]) // len(ref_data['num_atoms'])
                        ref_sync[prop] = ref_data[prop][:min_size * stress_per_structure]
                        pred_sync[prop] = pred_data[prop][:min_size * stress_per_structure]
                else:
                    # For scalar properties per structure
                    ref_sync[prop] = ref_data[prop][:min_size]
                    pred_sync[prop] = pred_data[prop][:min_size]
            else:
                ref_sync[prop] = ref_data[prop]
                pred_sync[prop] = pred_data[prop]
        
        if verbose:
            print(f"  Synchronized successfully!")
            for prop in ['energy', 'forces', 'stress']:
                if ref_sync[prop] is not None:
                    print(f"    {prop}: {len(ref_sync[prop])} elements")
        
        return ref_sync, pred_sync
    

def main():
    parser = argparse.ArgumentParser(
        description='Compare reference and predicted properties from XYZ files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare using combined file with REF and MACE prefixes
  python xyz_comparator.py --combined test-evaluated.xyz --ref-prefix REF --pred-prefix MACE

  # Compare using separate files
  python xyz_comparator.py --ref ref.xyz --pred mace.xyz --properties energy forces

  # Analyze file properties first
  python xyz_comparator.py --analyze test-evaluated.xyz

  # Save plots to file without showing
  python xyz_comparator.py --combined test-evaluated.xyz --output comparison.png --no-show
        """
    )
    
    # File input options
    file_group = parser.add_mutually_exclusive_group(required=True)
    file_group.add_argument('--combined', '-c', type=str, 
                           help='XYZ file containing both reference and predicted data')
    file_group.add_argument('--ref', type=str,
                           help='Reference XYZ file (use with --pred)')
    file_group.add_argument('--analyze', '-a', type=str,
                           help='Analyze properties in XYZ file and exit')
    
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode: compare combined vs separate loading')
    
    parser.add_argument('--pred', type=str,
                       help='Predicted XYZ file (use with --ref)')
    
    # Prefix options
    parser.add_argument('--ref-prefix', default='REF', type=str,
                       help='Prefix for reference properties (default: REF)')
    parser.add_argument('--pred-prefix', default='MACE', type=str,
                       help='Prefix for predicted properties (default: MACE)')
    
    # Comparison options
    parser.add_argument('--properties', '-p', nargs='+', 
                       choices=['energy', 'forces', 'stress'],
                       default=['energy', 'forces'],
                       help='Properties to compare (default: energy forces)')
    
    # Output options
    parser.add_argument('--output', '-o', type=str,
                       help='Save plots to this file (e.g., comparison.png)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots interactively')
    parser.add_argument('--sync', action='store_true',
                       help='Force synchronization of datasets (useful when comparing different sized files)')
    
    args = parser.parse_args()
    
    # Create comparator
    comparator = XYZComparator()
    
    # Handle analyze mode
    if args.analyze:
        comparator.analyze_xyz_properties(args.analyze)
        return
    
    # Handle debug mode
    if args.debug:
        # For debug, we need both combined and separate files
        if not (args.combined and os.path.exists("ref.xyz") and os.path.exists("mace.xyz")):
            print("Debug mode requires test-evaluated.xyz, ref.xyz, and mace.xyz files")
            sys.exit(1)
        
        comparator.debug_compare_data_loading(
            ref_xyz="ref.xyz",
            pred_xyz="mace.xyz", 
            combined_xyz=args.combined,
            ref_prefix=args.ref_prefix,
            pred_prefix=args.pred_prefix
        )
        return
    
    # Validate arguments
    if args.ref and not args.pred:
        parser.error("--pred is required when using --ref")
    if args.pred and not args.ref:
        parser.error("--ref is required when using --pred")
    
    # Check if files exist
    files_to_check = []
    if args.combined:
        files_to_check.append(args.combined)
    if args.ref:
        files_to_check.extend([args.ref, args.pred])
    
    for filepath in files_to_check:
        if not os.path.exists(filepath):
            print(f"Error: File '{filepath}' not found")
            sys.exit(1)
    
    # Run comparison
    try:
        comparator.compare_xyz_files(
            ref_xyz=args.ref,
            pred_xyz=args.pred,
            combined_xyz=args.combined,
            ref_prefix=args.ref_prefix,
            pred_prefix=args.pred_prefix,
            properties=args.properties,
            plot_path=args.output,
            show_plots=not args.no_show
        )
    except Exception as e:
        print(f"Error during comparison: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
