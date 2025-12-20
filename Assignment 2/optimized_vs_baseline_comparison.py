"""
Final Comparison of Optimized (Fine-Tuned) vs Baseline RoBERTa Models on IMDB dataset
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths:

optimized_csv = "test_results/train_val_test_full_comparison.csv"
baseline_csv  = "baseline_model_results/baseline_metrics_comparison.csv"

os.makedirs("final_comparison_plots", exist_ok=True)

# Load CSVs

opt_df  = pd.read_csv(optimized_csv)
base_df = pd.read_csv(baseline_csv)

base_df = base_df.rename(columns={"accuracy": "acc"})

# Metrics:

metrics = ["acc", "precision", "recall", "f1", "error", "loss", "runtime"]

# Print TEST set comparison:

print("\nTEST SET COMPARISON:")
for m in metrics:
    opt_val  = opt_df.loc[opt_df["split"]=="test", m].values[0]
    base_val = base_df.loc[base_df["split"]=="test", m].values[0]
    print(f"{m.upper():<10} | Optimized (Fine-Tuned): {opt_val:.4f} | Baseline RoBERTa: {base_val:.4f}")

# Bar plot: metrics comparison

plot_metrics = [m for m in metrics if m != "runtime"]
x = np.arange(len(plot_metrics))
width = 0.35

opt_vals  = [opt_df.loc[opt_df["split"]=="test", m].values[0] for m in plot_metrics]
base_vals = [base_df.loc[base_df["split"]=="test", m].values[0] for m in plot_metrics]

plt.figure(figsize=(12,6))
plt.bar(x - width/2, opt_vals, width, label="Optimized (Fine-Tuned)")
plt.bar(x + width/2, base_vals, width, label="Baseline RoBERTa")
plt.xticks(x, [m.upper() for m in plot_metrics])
plt.ylabel("Value")
plt.title("Test Set Metrics: Optimized (Fine-Tuned) vs Baseline RoBERTa")
plt.legend()
plt.savefig("final_comparison_plots/test_metrics_comparison.png", bbox_inches="tight")
plt.show()

# Runtime comparison:

opt_runtime  = opt_df.loc[opt_df["split"]=="test", "runtime"].values[0]
base_runtime = base_df.loc[base_df["split"]=="test", "runtime"].values[0]

plt.figure()
plt.bar(["Optimized (Fine-Tuned)","Baseline RoBERTa"], [opt_runtime, base_runtime])
plt.ylabel("Seconds")
plt.title("Test Set Runtime Comparison")
plt.savefig("final_comparison_plots/test_runtime_comparison.png", bbox_inches="tight")
plt.show()

# Loss curves (train / val / test):

splits = ["train", "val", "test"]

opt_loss  = [opt_df.loc[opt_df["split"]==s, "loss"].values[0] for s in splits]
base_loss = [base_df.loc[base_df["split"]==s, "loss"].values[0] for s in splits]

plt.figure()
plt.plot(splits, opt_loss, marker="o", label="Optimized")
plt.plot(splits, base_loss, marker="o", label="Baseline RoBERTa")
plt.ylabel("Cross-Entropy Loss")
plt.title("Loss Comparison: Optimized (Fine-Tuned) vs Baseline RoBERTa")
plt.legend()
plt.savefig("final_comparison_plots/loss_curves_comparison.png", bbox_inches="tight")
plt.show()
