#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

import argparse


def process_xyz_file(xyz_path):
    """Process an XYZ file to extract reference and MACE-predicted properties."""
    structures = read(xyz_path, ":", format="extxyz")
    e_ref, f_ref, s_ref = [], [], []
    e_pred, f_pred, s_pred = [], [], []

    for atoms in structures:
        n_atoms = len(atoms)

        # Extract reference energy per atom
        energy_ref = atoms.info.get("REF_energy")
        #e_ref.append(energy_ref / n_atoms)
        e_ref.append(energy_ref )

        # Extract MACE-predicted energy per atom
        energy_pred = atoms.info.get("MACE_energy")
        #e_pred.append(energy_pred / n_atoms)
        e_pred.append(energy_pred)

        # Extract reference forces
        if atoms.info.get("config_type") == "IsolatedAtom":
            f_ref.append(np.zeros((n_atoms, 3)).flatten())
        else:
            forces_ref = atoms.arrays.get("REF_forces") 
            f_ref.append(forces_ref.flatten() if forces_ref is not None else np.zeros((n_atoms, 3)).flatten())

        # Extract MACE-predicted forces
        forces_pred = atoms.arrays.get("MACE_forces")
        f_pred.append(forces_pred.flatten())

        # Extract reference stress
        if atoms.info.get("config_type") == "IsolatedAtom":
            s_ref.append(np.zeros((3, 3)).flatten())
        else:
            stress_ref = atoms.info.get("REF_stress")
            s_ref.append(np.fromstring(stress_ref, sep=" ") if stress_ref else np.zeros((3, 3)).flatten())

        # Extract MACE-predicted stress
        stress_pred = atoms.info.get("MACE_stress")
        if isinstance(stress_pred, str):
            s_pred.append(np.array(eval(stress_pred.replace("_JSON", ""))).flatten())
        else:
            s_pred.append(stress_pred.flatten())

    return (
        np.array(e_ref), np.array(e_pred),
        np.concatenate(f_ref) if f_ref else np.array([]),
        np.concatenate(f_pred) if f_pred else np.array([]),
        np.array(s_ref), np.array(s_pred)
    )


def calculate_mae_rmse(pred, ref):
    """Calculate MAE and RMSE between predictions and references."""
    mae = np.mean(np.abs(pred - ref))
    rmse = np.sqrt(np.mean((pred - ref) ** 2))
    return mae, rmse


def print_statistics(name, pred, ref):
    """Print MAE and RMSE statistics."""
    if len(pred) == 0 or len(ref) == 0:
        print(f"No data available for {name}")
        return None
    
    mae = np.mean(np.abs(pred - ref))
    rmse = np.sqrt(np.mean((pred - ref) ** 2))
    
    print(f"{name} Statistics:")
    print(f"  MAE:  {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    
    return {"mae": mae, "rmse": rmse}


def create_parity_plot(y_pred, y_ref, title, xlabel, ylabel, unit="", dataset_name=""):
    """Create a parity plot comparing predictions vs reference values."""
    if len(y_pred) == 0 or len(y_ref) == 0:
        print(f"Cannot create parity plot for {title}: no data")
        return None, None
    
    fig, ax = plt.subplots(figsize=(8, 8))
    mae = np.mean(np.abs(y_pred - y_ref))
    rmse = np.sqrt(np.mean((y_pred - y_ref)**2))
    
    ax.scatter(y_ref, y_pred, alpha=0.7, s=10)
    
    # Set axis limits
    min_val = min(np.min(y_ref), np.min(y_pred))
    max_val = max(np.max(y_ref), np.max(y_pred))
    margin = (max_val - min_val) * 0.05
    lims = [min_val - margin, max_val + margin]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # Add diagonal line
    ax.plot(lims, lims, 'k--', label='y=x')
    
    # Add statistics to the plot
    stats_text = f"MAE: {mae:.6f}\nRMSE: {rmse:.6f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set labels and title
    ax.set_xlabel(f"{xlabel} ({unit})")
    ax.set_ylabel(f"{ylabel} ({unit})")
    full_title = f"{title} - {dataset_name}" if dataset_name else title
    ax.set_title(full_title)
    
    fig.tight_layout()
    
    # Save figure with appropriate filename
    filename_base = title.split(':')[0].lower().replace(' ', '_')
    if dataset_name:
        save_path = f"{filename_base}_{dataset_name.lower()}.png"
    else:
        save_path = f"{filename_base}.png"
    
    fig.savefig(save_path, dpi=300)
    print(f"Saved parity plot to {save_path}")
    
    return fig, ax


def create_summary_figure(e_ref, e_pred, f_ref, f_pred, s_ref, s_pred, dataset_names):
    """Create a summary figure with 9 subplots for energy, forces, and stresses."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    properties = [(e_ref, e_pred, "Energy per atom (eV)"),
                  (f_ref, f_pred, "Forces (eV/Å)"),
                  (s_ref, s_pred, "Stresses (eV/Å³)")]

    markers = ['o', 's', '^']  # Different markers for datasets

    for row, (ref, pred, ylabel) in enumerate(properties):
        for col, dataset_name in enumerate(dataset_names):
            ax = axes[row, col]
            ax.scatter(ref[col], pred[col], alpha=0.7, s=10, label=f"{dataset_name} set", marker=markers[col])

            # Set axis limits
            min_val = min(np.min(ref[col]), np.min(pred[col]))
            max_val = max(np.max(ref[col]), np.max(pred[col]))
            margin = (max_val - min_val) * 0.05
            lims = [min_val - margin, max_val + margin]
            ax.set_xlim(lims)
            ax.set_ylim(lims)

            # Add diagonal line
            ax.plot(lims, lims, 'k--', label='y=x')

            # Set labels and title
            if row == 0:
                ax.set_title(f"{dataset_name} set")
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 2:
                ax.set_xlabel("Reference")

    fig.tight_layout()
    plt.savefig("summary_figure.png", dpi=300)
    print("Saved summary figure to summary_figure.png")
    plt.show()


def create_energy_summary_figure(e_ref, e_pred, dataset_names):
    """Create a summary figure for energy without dividing by the number of atoms."""
    fig, ax = plt.subplots(figsize=(8, 8))
    markers = ['o', 's', '^']  # Different markers for datasets

    for i, dataset_name in enumerate(dataset_names):
        ax.scatter(e_ref[i], e_pred[i], alpha=0.7, s=10, label=f"{dataset_name} set", marker=markers[i])

    # Set axis limits
    all_ref = np.concatenate(e_ref)
    all_pred = np.concatenate(e_pred)
    min_val = min(np.min(all_ref), np.min(all_pred))
    max_val = max(np.max(all_ref), np.max(all_pred))
    margin = (max_val - min_val) * 0.05
    lims = [min_val - margin, max_val + margin]
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Add diagonal line
    ax.plot(lims, lims, 'k--', label='y=x')

    # Set labels and title
    ax.set_title("Energy Summary (Total Energy)")
    ax.set_xlabel("Reference Energy (eV)")
    ax.set_ylabel("Predicted Energy (eV)")
    ax.legend()

    fig.tight_layout()
    plt.savefig("energy_summary_figure.png", dpi=300)
    print("Saved energy summary figure to energy_summary_figure.png")
    plt.show()


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

    e_ref_all, e_pred_all = [], []
    f_ref_all, f_pred_all = [], []
    s_ref_all, s_pred_all = [], []

    for name, path in datasets.items():
        print(f"\nProcessing {name} dataset from {path}...")

        # Extract reference and predicted data
        e_ref, e_pred, f_ref, f_pred, s_ref, s_pred = process_xyz_file(path)

        # Append data for plotting
        e_ref_all.append(e_ref)
        e_pred_all.append(e_pred)
        f_ref_all.append(f_ref)
        f_pred_all.append(f_pred)
        s_ref_all.append(s_ref)
        s_pred_all.append(s_pred)

        # Calculate MAE and RMSE for each property
        e_mae, e_rmse = calculate_mae_rmse(e_pred, e_ref)
        f_mae, f_rmse = calculate_mae_rmse(f_pred, f_ref)
        s_mae, s_rmse = calculate_mae_rmse(s_pred, s_ref)

        print(f"Energies - MAE: {e_mae:.6f}, RMSE: {e_rmse:.6f}")
        print(f"Forces - MAE: {f_mae:.6f}, RMSE: {f_rmse:.6f}")
        print(f"Stresses - MAE: {s_mae:.6f}, RMSE: {s_rmse:.6f}")

    # Create summary figure
    create_summary_figure(e_ref_all, e_pred_all, f_ref_all, f_pred_all, s_ref_all, s_pred_all, list(datasets.keys()))
    create_energy_summary_figure(e_ref_all, e_pred_all, list(datasets.keys()))

if __name__ == "__main__":
    main()