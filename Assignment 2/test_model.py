"""
Test the best optimzed trained model on IMDB test set.

This script:
- Loads the best trained model
- Computes metrics on the test set
- Computes test loss
- Compares test metrics with training and validation metrics
- Plots metrics, loss, runtime, and confusion matrix
- Plots ROC curve and attention heatmaps
"""

import os
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from torchviz import make_dot
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from transformers import RobertaModel
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Create output directories:

os.makedirs("test_results", exist_ok=True)
os.makedirs("test_plots", exist_ok=True)
os.makedirs("test_comparison_plots", exist_ok=True)
os.makedirs("test_visualizations", exist_ok=True)

# Model Definition (same as training):

class RobertaClassifier(nn.Module):
    """
    RoBERTa encoder with a lightweight classification head.
    """
    def __init__(self, dropout: float, pooling: str):
        super().__init__()
        self.pooling = pooling
        self.roberta = RobertaModel.from_pretrained("roberta-base", output_attentions=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        attentions = outputs.attentions  # tuple of layer attention tensors

        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            pooled = (hidden * attention_mask.unsqueeze(-1)).sum(1)
            pooled = pooled / attention_mask.sum(1, keepdim=True)

        logits = self.classifier(self.dropout(pooled))
        return logits, attentions  # return attention for visualization

# Metrics:

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    error = 1.0 - acc
    return acc, p, r, f1, error

# Load data:

def load_split(name):
    return (
        torch.load(f"tokenized_files/{name}_inputs.pt"),
        torch.load(f"tokenized_files/{name}_masks.pt"),
        torch.load(f"tokenized_files/{name}_labels.pt"),
    )

test_inputs, test_masks, test_labels = load_split("test")

# Load best config and metrics:

with open("best_config.json", "r") as f:
    best_config = json.load(f)

with open("best_metrics.json", "r") as f:
    best_metrics = json.load(f)

# Load train/val loss directly from .pt file:

loss_data = torch.load("best_model_loss.pt")
train_loss = loss_data["train_loss"].item()
val_loss = loss_data["val_loss"].item()

# Load model:

model = RobertaClassifier(
    dropout=best_config["dropout"],
    pooling=best_config["pooling"]
).to(DEVICE)

model.load_state_dict(torch.load("best_model.pt"))
model.eval()

loss_fn = nn.CrossEntropyLoss()

# Testing Loop:

test_loader = DataLoader(
    TensorDataset(test_inputs, test_masks, test_labels),
    batch_size=best_config["batch_size"],
)

test_preds = []
test_probs = []  # for ROC
test_loss_total = 0.0
start_time = time.time()
all_attentions = []

with torch.no_grad():
    for ids, masks, labels in test_loader:
        ids, masks, labels = ids.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE)
        logits, attentions = model(ids, masks)
        loss = loss_fn(logits, labels)

        test_loss_total += loss.item() * ids.size(0)
        test_preds.extend(logits.argmax(1).cpu().numpy())
        test_probs.extend(torch.softmax(logits, dim=1)[:,1].cpu().numpy())  # positive class prob
        all_attentions.append([a.cpu() for a in attentions])  # save attention

test_runtime = time.time() - start_time
test_loss = test_loss_total / len(test_loader.dataset)

print(f"Test runtime: {test_runtime:.2f}s")
print(f"Test loss: {test_loss:.4f}")

# Compute test metrics:

acc, p, r, f1, err = compute_metrics(test_labels.numpy(), test_preds)

# Save test predictions:

torch.save(test_preds, "test_results/test_predictions.pt")

# Save metrics comparison CSV:

comparison_df = pd.DataFrame([
    {
        "split": "train",
        "acc": best_metrics["train_acc"],
        "precision": best_metrics["train_precision"],
        "recall": best_metrics["train_recall"],
        "f1": best_metrics["train_f1"],
        "error": best_metrics["train_error"],
        "loss": train_loss,
        "runtime": best_metrics["runtime"],
    },
    {
        "split": "val",
        "acc": best_metrics["val_acc"],
        "precision": best_metrics["val_precision"],
        "recall": best_metrics["val_recall"],
        "f1": best_metrics["val_f1"],
        "error": best_metrics["val_error"],
        "loss": val_loss,
        "runtime": best_metrics["runtime"],
    },
    {
        "split": "test",
        "acc": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "error": err,
        "loss": test_loss,
        "runtime": test_runtime,
    }
])

comparison_df.to_csv(
    "test_results/train_val_test_full_comparison.csv",
    index=False
)

print("\nTrain / Val / Test Comparison:")
print(comparison_df)

# Plot metrics comparison:

metrics = ["acc", "precision", "recall", "f1", "error", "loss"]
x = np.arange(len(metrics))
width = 0.25

plt.figure(figsize=(12,6))
plt.bar(x - width, comparison_df.iloc[0][metrics], width, label="Train")
plt.bar(x, comparison_df.iloc[1][metrics], width, label="Val")
plt.bar(x + width, comparison_df.iloc[2][metrics], width, label="Test")
plt.xticks(x, [m.upper() for m in metrics])
plt.ylabel("Value")
plt.title("Train vs Val vs Test Metrics & Loss")
plt.legend()
plt.savefig("test_comparison_plots/train_val_test_metrics_loss.png", bbox_inches="tight")
plt.show()

# Runtime plot:

plt.figure()
plt.bar(comparison_df["split"], comparison_df["runtime"])
plt.ylabel("Seconds")
plt.title("Runtime Comparison")
plt.savefig("test_comparison_plots/runtime_comparison.png", bbox_inches="tight")
plt.show()

# Confusion Matrix (Test):

cm = confusion_matrix(test_labels.numpy(), test_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
disp.plot(cmap="Blues")
plt.title("Test Confusion Matrix")
plt.savefig("test_plots/test_confusion_matrix.png", bbox_inches="tight")
plt.show()

# Plot train/val/test loss curves:

plt.figure()
plt.plot(["Train","Val","Test"], [train_loss,val_loss,test_loss], marker='o')
plt.title("Train/Val/Test Loss Comparison")
plt.ylabel("Cross-Entropy Loss")
plt.savefig("test_plots/train_val_test_loss_curve.png", bbox_inches="tight")
plt.show()

# ROC Curve for Test Set:

fpr, tpr, thresholds = roc_curve(test_labels.numpy(), test_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC={roc_auc:.4f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Test Set ROC Curve")
plt.legend(loc="lower right")
plt.savefig("test_plots/test_roc_curve.png", bbox_inches="tight")
plt.show()

# Attention Visualization Heatmap (last layer, first head, first sample):

sample_attention = all_attentions[-1][-1][0,0,:,:].numpy()  # last batch, last layer, first head
plt.figure(figsize=(10,8))
plt.imshow(sample_attention, cmap='viridis', aspect='auto')
plt.colorbar(label="Attention Score")
plt.title("Attention Heatmap (Last Layer, Head 0, First Sample of Last Batch)")
plt.xlabel("Token Position")
plt.ylabel("Token Position")
plt.savefig("test_plots/attention_heatmap.png", bbox_inches="tight")
plt.show()

# Graphviz Model Visualization:

dummy_ids = test_inputs[:1].to(DEVICE)
dummy_masks = test_masks[:1].to(DEVICE)
dummy_output,_ = model(dummy_ids, dummy_masks)

dot = make_dot(dummy_output, params=dict(model.named_parameters()))
dot.render("test_visualizations/roberta_classifier_graph", format="pdf")
