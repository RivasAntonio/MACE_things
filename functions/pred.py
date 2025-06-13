#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import argparse
from sklearn.metrics import r2_score


def process_xyz_file(xyz_path):
    """Process an XYZ file to extract reference and MACE-predicted properties."""
    structures = read(xyz_path, ":", format="extxyz")
    e_atom_ref, f_ref, s_ref, e_atom_pred, f_pred, s_pred = [], [], [], [], [], []

    for atoms in structures:
        if atoms.info.get("config_type") == "IsolatedAtom":
            continue

        n_atoms = len(atoms)

        # Extract reference energies per atom
        energy_ref = atoms.info.get("REF_energy")
        e_atom_ref.append(energy_ref / n_atoms)
        
        # Extract MACE-predicted energies per atom
        energy_pred = atoms.info.get("MACE_energy")
        e_atom_pred.append(energy_pred / n_atoms)
        
        # Extract reference forces
        forces_ref = atoms.arrays.get("REF_forces") 
        f_ref.append(forces_ref.flatten())

        # Extract MACE-predicted forces
        forces_pred = atoms.arrays.get("MACE_forces")
        f_pred.append(forces_pred.flatten())

        # Extract reference stress
        stress_ref = atoms.info.get("REF_stress")
        s_ref.append(stress_ref)

        # Extract MACE-predicted stress
        stress_pred = atoms.info.get("MACE_stress")
        s_pred.append(np.array(stress_pred).flatten())

    return (
        np.array(e_atom_ref), np.array(e_atom_pred),
        np.concatenate(f_ref) if f_ref else np.array([]),
        np.concatenate(f_pred) if f_pred else np.array([]),
        np.array(s_ref), np.array(s_pred)
    )


def calculate_mae_rmse(pred, ref):
    """Calculate MAE and RMSE between predictions and references."""
    mae = np.mean(np.abs(pred - ref))
    rmse = np.sqrt(np.mean((pred - ref) ** 2))
    return mae, rmse

#def print_statistics(name, pred, ref):
#    """Print MAE and RMSE statistics."""
#    if len(pred) == 0 or len(ref) == 0:
#        print(f"No data available for {name}")
#        return None
#    
#    pred = np.concatenate(pred) if isinstance(pred, list) else np.array(pred)
#    ref = np.concatenate(ref) if isinstance(ref, list) else np.array(ref)
#    mae = np.mean(np.abs(pred.flatten() - ref.flatten()))
#    rmse = np.sqrt(np.mean((pred.flatten() - ref.flatten()) ** 2))
#    
#    print(f"{name} Statistics:")
#    print(f"  MAE:  {mae:.6f}")
#    print(f"  RMSE: {rmse:.6f}")
#    
#    return {"mae": mae, "rmse": rmse}


def plot_evaluation(e_atom_ref_all, e_atom_pred_all, f_ref_all, f_pred_all, s_ref_all, s_pred_all, dataset_names):
    """Plot evaluation figures including a single summary figure."""
    # Filter out values where any property is less than -11
    def filter_data(ref_all, pred_all):
        filtered_ref_all, filtered_pred_all = [], []
        for ref, pred in zip(ref_all, pred_all):
            mask = ref >= -11
            filtered_ref_all.append(ref[mask])
            filtered_pred_all.append(pred[mask])
        return np.array(filtered_ref_all), np.array(filtered_pred_all)

    e_atom_ref_all, e_atom_pred_all = filter_data(e_atom_ref_all, e_atom_pred_all)
    f_ref_all, f_pred_all = filter_data(f_ref_all, f_pred_all)
    s_ref_all, s_pred_all = filter_data(s_ref_all, s_pred_all)

    # Create a single summary figure with 1 row and 3 columns for energy per atom, forces, and stresses
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    properties = [
        (e_atom_ref_all, e_atom_pred_all, "Energy per Atom (eV)"),
        (f_ref_all, f_pred_all, "Forces (eV/Å)"),
        (s_ref_all, s_pred_all, "Stresses (eV/Å³)")
    ]
    colors = ['blue', 'orange', 'green']
    markers = ['o', 's', '^']

    for col, (ref_all, pred_all, ylabel) in enumerate(properties):
        ax = axes[col]
        for i, dataset_name in enumerate(dataset_names):
            # Downsample data to plot every 100 points
            ref_sampled = ref_all[i][::100]
            pred_sampled = pred_all[i][::100]
            ax.scatter(ref_sampled, pred_sampled, alpha=0.7, s=10, 
                       label=f"{dataset_name} set", color=colors[i % len(colors)], marker=markers[i % len(markers)])

        # Set axis limits
        all_ref = np.concatenate(ref_all)
        all_pred = np.concatenate(pred_all)
        min_val, max_val = min(all_ref.min(), all_pred.min()), max(all_ref.max(), all_pred.max())
        margin = (max_val - min_val) * 0.05
        lims = [min_val - margin, max_val + margin]
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # Add diagonal line
        ax.plot(lims, lims, 'k--', label='y=x')

        # Calculate R²
        r2 = r2_score(all_ref, all_pred)

        # Add RMSE and R² to the plot
        ax.text(0.05, 0.95, f"R²: {r2:.8f}", transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # Set labels and title
        ax.set_title(ylabel)
        ax.set_xlabel("Reference")
        if col == 0:
            ax.set_ylabel("Predicted")

        ax.legend()

    fig.tight_layout()
    filename = "model_evaluation.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved summary figure to {filename}")


# --- MAIN ---
def main():
    parser = argparse.ArgumentParser(description="Evaluate MACE predictions against reference data.")
    parser.add_argument('--train', required=True, help="Path to the train XYZ file.")
    parser.add_argument('--validation', required=True, help="Path to the validation XYZ file.")
    parser.add_argument('--test', required=True, help="Path to the test XYZ file.")
    args = parser.parse_args()

    datasets = {
        'train': args.train,
        'validation': args.validation,
        'test': args.test
    }

    e_atom_ref_all, e_atom_pred_all, f_ref_all, f_pred_all, s_ref_all, s_pred_all = [], [], [], [], [], []    

    for name, path in datasets.items():
        print(f"\nProcessing {name} dataset from {path}...")

        # Extract reference and predicted data
        e_atom_ref, e_atom_pred, f_ref, f_pred, s_ref, s_pred = process_xyz_file(path)

        # Append data for plotting
        e_atom_pred_all =+ e_atom_pred
        e_atom_ref_all =+ e_atom_ref
        f_ref_all =+ f_ref
        f_pred_all =+ f_pred
        s_ref_all =+ s_ref
        s_pred_all =+ s_pred

        #e_atom_ref_all.append(e_atom_ref)
        #e_atom_pred_all.append(e_atom_pred)
        #f_ref_all.append(f_ref)
        #f_pred_all.append(f_pred)
        #s_ref_all.append(s_ref)
        #s_pred_all.append(s_pred)

    print(e_atom_pred_all)
    print(f_pred_all)
    print(s_pred_all)
    print(e_atom_ref_all)
    print(f_ref_all)
    print(s_ref_all)

    # Calculate MAE and RMSE for each property
    e_mae, e_rmse = calculate_mae_rmse(np.array(e_atom_pred_all), np.array(e_atom_ref_all))
    f_mae, f_rmse = calculate_mae_rmse(np.array(f_pred_all), np.array(f_ref_all))
    s_mae, s_rmse = calculate_mae_rmse(np.array(s_pred_all), np.array(s_ref_all))
    print(f"Energies - MAE: {e_mae:.6f}, RMSE: {e_rmse:.6f}")
    print(f"Forces - MAE: {f_mae:.6f}, RMSE: {f_rmse:.6f}")
    print(f"Stresses - MAE: {s_mae:.6f}, RMSE: {s_rmse:.6f}")


    # Create summary figures
#    plot_evaluation(e_atom_ref_all, e_atom_pred_all, f_ref_all, f_pred_all, s_ref_all, s_pred_all, list(datasets.keys()))

if __name__ == "__main__":
    main()
