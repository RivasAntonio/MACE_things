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

    # --------------------------
    # Energy-related metrics
    # --------------------------
    plt.figure(figsize=(14, 10))
    plt.suptitle(f"Energy Metrics (from epoch {min_epoch})", y=1.02, fontsize=16)
    
    # MAE Energy
    plt.subplot(2, 2, 1)
    plt.plot(epochs, [d["mae_e"] for d in eval_data], 'b-o')
    plt.title("MAE Energy (eV)")
    plt.xlabel("Epochs")
    plt.ylabel("MAE E (eV)")
    plt.grid(True)
    
    # MAE Energy per atom
    plt.subplot(2, 2, 2)
    plt.plot(epochs, [d["mae_e_per_atom"] for d in eval_data], 'b-o')
    plt.title("MAE Energy per Atom (eV/atom)")
    plt.xlabel("Epochs")
    plt.ylabel("MAE E (eV/atom)")
    plt.grid(True)
    
    # RMSE Energy
    plt.subplot(2, 2, 3)
    plt.plot(epochs, [d["rmse_e"] for d in eval_data], 'r-o')
    plt.title("RMSE Energy (eV)")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE E (eV)")
    plt.grid(True)
    
    # RMSE Energy per atom
    plt.subplot(2, 2, 4)
    plt.plot(epochs, [d["rmse_e_per_atom"] for d in eval_data], 'r-o')
    plt.title("RMSE Energy per Atom (eV/atom)")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE E (eV/atom)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # --------------------------
    # Force-related metrics
    # --------------------------
    plt.figure(figsize=(14, 10))
    plt.suptitle(f"Force Metrics (from epoch {min_epoch})", y=1.02, fontsize=16)
    
    # MAE Forces
    plt.subplot(2, 2, 1)
    plt.plot(epochs, [d["mae_f"] for d in eval_data], 'g-o')
    plt.title("MAE Forces (eV/Å)")
    plt.xlabel("Epochs")
    plt.ylabel("MAE F (eV/Å)")
    plt.grid(True)
    
    # Relative MAE Forces
    plt.subplot(2, 2, 2)
    plt.plot(epochs, [d["rel_mae_f"] for d in eval_data], 'g-o')
    plt.title("Relative MAE Forces (%)")
    plt.xlabel("Epochs")
    plt.ylabel("Relative MAE F (%)")
    plt.grid(True)
    
    # RMSE Forces
    plt.subplot(2, 2, 3)
    plt.plot(epochs, [d["rmse_f"] for d in eval_data], 'm-o')
    plt.title("RMSE Forces (eV/Å)")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE F (eV/Å)")
    plt.grid(True)
    
    # Relative RMSE Forces
    plt.subplot(2, 2, 4)
    plt.plot(epochs, [d["rel_rmse_f"] for d in eval_data], 'm-o')
    plt.title("Relative RMSE Forces (%)")
    plt.xlabel("Epochs")
    plt.ylabel("Relative RMSE F (%)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # --------------------------
    # Additional metrics
    # --------------------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    plt.suptitle(f"Additional Metrics (from epoch {min_epoch})", y=1.05, fontsize=16)
    
    axs[0].plot(epochs, [d["q95_e"] for d in eval_data], 'c-o', label="Q95 E")
    axs[0].set_title("95th Percentile Energy Error (eV)")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Q95 E (eV)")
    axs[0].grid(True)
    
    axs[1].plot(epochs, [d["time"] for d in eval_data], 'k-o', label="Eval Time")
    axs[1].set_title("Evaluation Time")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Time (s)")
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()

    # --------------------------
    # Optimization plots
    # --------------------------
    if opt_by_epoch:
        # Filter optimization data by min_epoch
        opt_epochs = sorted([e for e in opt_by_epoch.keys() if e >= min_epoch])
        
        if opt_epochs:
            avg_loss_per_epoch = [np.mean([x["loss"] for x in opt_by_epoch[epoch]]) for epoch in opt_epochs]
            total_time_per_epoch = [np.sum([x["time"] for x in opt_by_epoch[epoch]]) for epoch in opt_epochs]
            
            # Plot optimization loss
            plt.figure(figsize=(12, 5))
            plt.plot(opt_epochs, avg_loss_per_epoch, 'r-o', label="Avg Loss")
            plt.title(f"Optimization Loss (from epoch {min_epoch})")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.yscale('log')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            # Plot optimization time
            plt.figure(figsize=(12, 5))
            plt.plot(opt_epochs, total_time_per_epoch, 'b-o', label="Total Time")
            plt.title(f"Optimization Time (from epoch {min_epoch})")
            plt.xlabel("Epochs")
            plt.ylabel("Time (s)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_mlff_training.py file.json [min_epoch]")
        sys.exit(1)

    file_path = sys.argv[1]
    min_epoch = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    eval_data, opt_by_epoch = load_jsonl(file_path)
    plot_training(eval_data, opt_by_epoch, min_epoch)