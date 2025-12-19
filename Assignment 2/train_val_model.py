"""
Train and validate an optimized RoBERTa-based sentiment classifier on IMDB.

Primary evaluation metric:
- F1-score (binary classification)

Target metric:
- Validation F1 >= 0.90

This script performs:
- Hyperparameter search using Randomized Search
- Loss function used: Cross-Entropy Loss
- Training and validation loops
- Early stopping
- Gradient clipping
- Runtime tracking
- Best model selection
"""

import os
import time
import itertools
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from transformers import RobertaModel
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Create output directories:

os.makedirs("train_val_results", exist_ok=True)
os.makedirs("train_val_plots", exist_ok=True)

# Fixed training settings:

MAX_EPOCHS = 10    # Upper bound
EARLY_STOPPING_PATIENCE = 2
MAX_GRAD_NORM = 1.0     # for gradient clipping

# Model:

class RobertaClassifier(nn.Module):
    """
    RoBERTa encoder with a lightweight classification head.

    Pooling strategies:
    - 'cls': use <s> token representation
    - 'mean': mean pooling over valid tokens
    """

    def __init__(self, dropout: float, pooling: str):
        super().__init__()
        self.pooling = pooling
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state

        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            pooled = (hidden * attention_mask.unsqueeze(-1)).sum(1)
            pooled = pooled / attention_mask.sum(1, keepdim=True)

        return self.classifier(self.dropout(pooled))

# Metrics:

def compute_metrics(y_true, y_pred):
    """
    Compute accuracy, precision, recall, F1-score, and error.
    """
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    error = 1.0 - acc
    return acc, p, r, f1, error

# for loading data:

def load_split(name: str):
    """
    Load tokenized tensors for a given split.
    """
    return (
        torch.load(f"tokenized_files/{name}_inputs.pt"),
        torch.load(f"tokenized_files/{name}_masks.pt"),
        torch.load(f"tokenized_files/{name}_labels.pt"),
    )

train_inputs, train_masks, train_labels = load_split("train")
val_inputs, val_masks, val_labels = load_split("val")

# Hyperparameter grid:

hyperparam_grid = {
    "lr": [1e-5, 2e-5, 3e-5],
    "batch_size": [16, 32],
    "dropout": [0.1, 0.2],
    "weight_decay": [0.0, 0.01],
    "scheduler": ["linear", "cosine"],
    "warmup_ratio": [0.06, 0.1], 
    "pooling": ["cls", "mean"],
}

# Randomized Search Setup:

NUM_RANDOM_CONFIGS = 8

all_configs = list(itertools.product(*hyperparam_grid.values()))
random_configs = random.sample(all_configs, NUM_RANDOM_CONFIGS)

# Training and Validation:

results = []
best_f1 = 0.0
best_model_state = None
best_config = None
best_val_preds = None
best_val_labels = None
best_metrics = None

for values in random_configs:
    config = dict(zip(hyperparam_grid.keys(), values))
    print("\n==============================")
    print("Running config:", config)
    print("==============================")

    train_loader = DataLoader(
        TensorDataset(train_inputs, train_masks, train_labels),
        batch_size=config["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_inputs, val_masks, val_labels),
        batch_size=config["batch_size"],
    )

    model = RobertaClassifier(
        dropout=config["dropout"],
        pooling=config["pooling"],
    ).to(DEVICE)

    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    total_steps = len(train_loader) * MAX_EPOCHS
    warmup_steps = int(config["warmup_ratio"] * total_steps)

    scheduler = (
        get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        if config["scheduler"] == "linear"
        else get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    )

    loss_fn = nn.CrossEntropyLoss()

    best_epoch_f1 = 0.0
    patience = 0
    start_time = time.time()

    for epoch in range(MAX_EPOCHS):
        # Training
        model.train()
        train_preds, train_labels_all = [], []

        for ids, masks, labels in train_loader:
            ids, masks, labels = ids.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(ids, masks)
            loss = loss_fn(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

            optimizer.step()
            scheduler.step()

            train_preds.extend(logits.argmax(1).cpu().numpy())
            train_labels_all.extend(labels.cpu().numpy())

        train_acc, train_p, train_r, train_f1, train_err = compute_metrics(
            train_labels_all, train_preds
        )

        # Validation
        model.eval()
        val_preds, val_labels_all = [], []

        with torch.no_grad():
            for ids, masks, labels in val_loader:
                ids, masks = ids.to(DEVICE), masks.to(DEVICE)
                logits = model(ids, masks)
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_labels_all.extend(labels.numpy())

        val_acc, val_p, val_r, val_f1, val_err = compute_metrics(
            val_labels_all, val_preds
        )

        print(
            f"Epoch {epoch+1} | "
            f"Train Acc: {train_acc:.4f} P: {train_p:.4f} R: {train_r:.4f} "
            f"F1: {train_f1:.4f} Err: {train_err:.4f} | "
            f"Val Acc: {val_acc:.4f} P: {val_p:.4f} R: {val_r:.4f} "
            f"F1: {val_f1:.4f} Err: {val_err:.4f}"
        )

        if val_f1 > best_epoch_f1:
            best_epoch_f1 = val_f1
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered")
                break

    runtime = time.time() - start_time

    results.append({
        **config,
        "train_acc": train_acc,
        "train_precision": train_p,
        "train_recall": train_r,
        "train_f1": train_f1,
        "train_error": train_err,
        "val_acc": val_acc,
        "val_precision": val_p,
        "val_recall": val_r,
        "val_f1": val_f1,
        "val_error": val_err,
        "runtime": runtime,
    })

    if best_epoch_f1 > best_f1:
        best_f1 = best_epoch_f1
        best_model_state = model.state_dict()
        best_config = config
        best_val_preds = val_preds
        best_val_labels = val_labels_all
        best_metrics = results[-1]

# Save best model & config & metrics:

torch.save(best_model_state, "best_model.pt")

with open("best_config.json", "w") as f:
    json.dump(best_config, f, indent=4)

with open("best_metrics.json", "w") as f:
    json.dump(best_metrics, f, indent=4)

pd.DataFrame(results).to_csv(
    "train_val_results/all_results.csv",
    index=False,
)

print("\nBEST CONFIG:", best_config)
print("Best Metrics:", best_metrics)

# Confusion Matrix (Validation):

cm = confusion_matrix(best_val_labels, best_val_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
disp.plot(cmap="Blues")
plt.title("Validation Confusion Matrix (Best Model)")
plt.savefig("train_val_plots/val_confusion_matrix.png", bbox_inches="tight")
plt.show()

# Plots:

metrics_to_plot = [
    "train_acc","val_acc",
    "train_precision","val_precision",
    "train_recall","val_recall",
    "train_f1","val_f1",
    "train_error","val_error"
]

for m in metrics_to_plot:
    plt.figure()
    plt.plot([r[m] for r in results])
    plt.title(m.replace("_"," ").title())
    plt.ylabel(m.split("_")[0].title())
    plt.savefig(f"train_val_plots/{m}.png", bbox_inches="tight")
    plt.show()

plt.figure()
plt.plot([r["runtime"] for r in results])
plt.title("Runtime per configuration")
plt.ylabel("Seconds")
plt.savefig("train_val_plots/runtime.png", bbox_inches="tight")
plt.show()
