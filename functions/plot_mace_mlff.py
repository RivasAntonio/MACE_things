#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_jsonl(file_path):
    """Load data from a JSONL file and separate evaluation from optimization."""
    eval_data = []
    opt_data = []
    
    with open(file_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            if entry["mode"] == "eval":
                eval_data.append(entry)
            elif entry["mode"] == "opt":
                opt_data.append(entry)
    
    # Sort eval_data by epoch if epoch information is available
    if eval_data and "epoch" in eval_data[0] and eval_data[0]["epoch"] is not None:
        eval_data.sort(key=lambda x: x["epoch"])
    
    # Group optimization data by epoch
    opt_by_epoch = defaultdict(list)
    for entry in opt_data:
        if "epoch" in entry:
            opt_by_epoch[entry["epoch"]].append(entry)
    
    return eval_data, opt_by_epoch

def filter_data_by_min_epoch(data, min_epoch=0, has_epoch_field=True):
    """Filter data to include only entries with epoch >= min_epoch"""
    if has_epoch_field:
        return [d for d in data if d["epoch"] is not None and d["epoch"] >= min_epoch]
    else:
        return data[min_epoch:] if min_epoch < len(data) else []

def plot_training(eval_data, opt_by_epoch, min_epoch=0):
    """Generate plots of all important metrics over epochs."""
    if not eval_data:
        print("No evaluation data to plot.")
        return
    
    # Determine if data has epoch field
    has_epoch_field = "epoch" in eval_data[0] and eval_data[0]["epoch"] is not None
    
    # Filter data based on min_epoch
    eval_data = filter_data_by_min_epoch(eval_data, min_epoch, has_epoch_field)
    
    if not eval_data:
        print(f"No data to plot after filtering for min_epoch={min_epoch}")
        return
    
    # Skip the first evaluation point if more than one remains after filtering
    if len(eval_data) > 1:
        eval_data = eval_data[1:]
    
    # Determine epochs
    if has_epoch_field:
        epochs = [d["epoch"] for d in eval_data]
    else:
        epochs = list(range(min_epoch, min_epoch + len(eval_data)))

    # --- Superfigura 1: Energía ---
    fig_energy, axs_energy = plt.subplots(2, 2, figsize=(10, 10))
    fig_energy.suptitle(f"Energy Metrics (from epoch {min_epoch})", y=1.02, fontsize=16)
    axs_energy[0, 0].plot(epochs, [d["mae_e"] for d in eval_data], 'b-o')
    axs_energy[0, 0].set_title("MAE Energy (eV)")
    axs_energy[0, 0].set_xlabel("Epochs")
    axs_energy[0, 0].set_ylabel("MAE E (eV)")
    axs_energy[0, 0].grid(True)
    axs_energy[0, 1].plot(epochs, [d["mae_e_per_atom"] for d in eval_data], 'b-o')
    axs_energy[0, 1].set_title("MAE Energy per Atom (eV/atom)")
    axs_energy[0, 1].set_xlabel("Epochs")
    axs_energy[0, 1].set_ylabel("MAE E (eV/atom)")
    axs_energy[0, 1].grid(True)
    axs_energy[1, 0].plot(epochs, [d["rmse_e"] for d in eval_data], 'r-o')
    axs_energy[1, 0].set_title("RMSE Energy (eV)")
    axs_energy[1, 0].set_xlabel("Epochs")
    axs_energy[1, 0].set_ylabel("RMSE E (eV)")
    axs_energy[1, 0].grid(True)
    axs_energy[1, 1].plot(epochs, [d["rmse_e_per_atom"] for d in eval_data], 'r-o')
    axs_energy[1, 1].set_title("RMSE Energy per Atom (eV/atom)")
    axs_energy[1, 1].set_xlabel("Epochs")
    axs_energy[1, 1].set_ylabel("RMSE E (eV/atom)")
    axs_energy[1, 1].grid(True)
    fig_energy.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    # --- Superfigura 2: Fuerzas ---
    fig_forces, axs_forces = plt.subplots(2, 2, figsize=(10, 10))
    fig_forces.suptitle(f"Force Metrics (from epoch {min_epoch})", y=1.02, fontsize=16)
    axs_forces[0, 0].plot(epochs, [d["mae_f"] for d in eval_data], 'g-o')
    axs_forces[0, 0].set_title("MAE Forces (eV/Å)")
    axs_forces[0, 0].set_xlabel("Epochs")
    axs_forces[0, 0].set_ylabel("MAE F (eV/Å)")
    axs_forces[0, 0].grid(True)
    axs_forces[0, 1].plot(epochs, [d["rel_mae_f"] for d in eval_data], 'g-o')
    axs_forces[0, 1].set_title("Relative MAE Forces (%)")
    axs_forces[0, 1].set_xlabel("Epochs")
    axs_forces[0, 1].set_ylabel("Relative MAE F (%)")
    axs_forces[0, 1].grid(True)
    axs_forces[1, 0].plot(epochs, [d["rmse_f"] for d in eval_data], 'm-o')
    axs_forces[1, 0].set_title("RMSE Forces (eV/Å)")
    axs_forces[1, 0].set_xlabel("Epochs")
    axs_forces[1, 0].set_ylabel("RMSE F (eV/Å)")
    axs_forces[1, 0].grid(True)
    axs_forces[1, 1].plot(epochs, [d["rel_rmse_f"] for d in eval_data], 'm-o')
    axs_forces[1, 1].set_title("Relative RMSE Forces (%)")
    axs_forces[1, 1].set_xlabel("Epochs")
    axs_forces[1, 1].set_ylabel("Relative RMSE F (%)")
    axs_forces[1, 1].grid(True)
    fig_forces.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    # --- Superfigura 3: Métricas adicionales ---
    fig_add, axs_add = plt.subplots(1, 2, figsize=(12, 5))
    fig_add.suptitle(f"Additional Metrics (from epoch {min_epoch})", y=1.05, fontsize=16)
    axs_add[0].plot(epochs, [d["q95_e"] for d in eval_data], 'c-o', label="Q95 E")
    axs_add[0].set_title("95th Percentile Energy Error (eV)")
    axs_add[0].set_xlabel("Epochs")
    axs_add[0].set_ylabel("Q95 E (eV)")
    axs_add[0].grid(True)
    axs_add[1].plot(epochs, [d["time"] for d in eval_data], 'k-o', label="Eval Time")
    axs_add[1].set_title("Evaluation Time")
    axs_add[1].set_xlabel("Epochs")
    axs_add[1].set_ylabel("Time (s)")
    axs_add[1].grid(True)
    fig_add.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # --- Superfigura 4: Optimización ---
    if opt_by_epoch:
        opt_epochs = sorted([e for e in opt_by_epoch.keys() if e >= min_epoch])
        if opt_epochs:
            avg_loss_per_epoch = [np.mean([x["loss"] for x in opt_by_epoch[epoch]]) for epoch in opt_epochs]
            total_time_per_epoch = [np.sum([x["time"] for x in opt_by_epoch[epoch]]) for epoch in opt_epochs]
            fig_opt, axs_opt = plt.subplots(1, 2, figsize=(14, 5))
            fig_opt.suptitle(f"Optimization Metrics (from epoch {min_epoch})", y=1.05, fontsize=16)
            axs_opt[0].plot(opt_epochs, avg_loss_per_epoch, 'r-o', label="Avg Loss")
            axs_opt[0].set_title("Optimization Loss")
            axs_opt[0].set_xlabel("Epochs")
            axs_opt[0].set_ylabel("Loss")
            axs_opt[0].set_yscale('log')
            axs_opt[0].grid(True)
            axs_opt[0].legend()
            axs_opt[1].plot(opt_epochs, total_time_per_epoch, 'b-o', label="Total Time")
            axs_opt[1].set_title("Optimization Time")
            axs_opt[1].set_xlabel("Epochs")
            axs_opt[1].set_ylabel("Time (s)")
            axs_opt[1].grid(True)
            axs_opt[1].legend()
            fig_opt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_mlff_training.py file.json [min_epoch]")
        sys.exit(1)

    file_path = sys.argv[1]
    min_epoch = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    eval_data, opt_by_epoch = load_jsonl(file_path)
    plot_training(eval_data, opt_by_epoch, min_epoch)