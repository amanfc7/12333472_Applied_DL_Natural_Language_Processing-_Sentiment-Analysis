"""
Baseline evaluation using off-the-shelf RoBERTa for IMDB sentiment classification.

This script:
- Uses the pretrained RoBERTa model with a default classification head
- Does NOT train the model
- Computes metrics on train, val, and test sets
- Plots metrics and confusion matrices
- Prints sample predictions vs true labels
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from transformers import RobertaModel

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


# --------------------------------------
# Silence TensorFlow warnings
# --------------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

BATCH_SIZE = 16  # baseline batch size

# --------------------------------------
# Baseline Model Definition
# --------------------------------------
class BaselineRobertaClassifier(nn.Module):
    """
    Baseline RoBERTa classifier.

    - Uses pretrained RoBERTa encoder
    - Classification head is randomly initialized
    - No training / fine-tuning
    """
    def __init__(self, pooling="cls"):
        super().__init__()
        self.pooling = pooling
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(0.1)
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

# --------------------------------------
# Metrics
# --------------------------------------
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    error = 1.0 - acc
    return acc, p, r, f1, error

# --------------------------------------
# Load tokenized data
# --------------------------------------
def load_split(name: str):
    return (
        torch.load(f"tokenized_files/{name}_inputs.pt"),
        torch.load(f"tokenized_files/{name}_masks.pt"),
        torch.load(f"tokenized_files/{name}_labels.pt"),
    )

train_inputs, train_masks, train_labels = load_split("train")
val_inputs, val_masks, val_labels = load_split("val")
test_inputs, test_masks, test_labels = load_split("test")

# --------------------------------------
# Initialize baseline model
# --------------------------------------
baseline_model = BaselineRobertaClassifier(pooling="cls").to(DEVICE)
baseline_model.eval()  # no training

# --------------------------------------
# Evaluation function
# --------------------------------------
def evaluate_model(model, inputs, masks, labels, batch_size=BATCH_SIZE):
    loader = DataLoader(
        TensorDataset(inputs, masks),
        batch_size=batch_size,
    )

    preds = []
    with torch.no_grad():
        for ids, mask in loader:
            ids, mask = ids.to(DEVICE), mask.to(DEVICE)
            logits = model(ids, mask)
            preds.extend(logits.argmax(1).cpu().numpy())

    acc, p, r, f1, err = compute_metrics(labels.numpy(), preds)
    return acc, p, r, f1, err, preds

# --------------------------------------
# Evaluate on Train / Val / Test
# --------------------------------------
train_acc, train_p, train_r, train_f1, train_err, train_preds = evaluate_model(
    baseline_model, train_inputs, train_masks, train_labels
)

val_acc, val_p, val_r, val_f1, val_err, val_preds = evaluate_model(
    baseline_model, val_inputs, val_masks, val_labels
)

test_acc, test_p, test_r, test_f1, test_err, test_preds = evaluate_model(
    baseline_model, test_inputs, test_masks, test_labels
)

# --------------------------------------
# Print metrics
# --------------------------------------
metrics_names = ["Accuracy", "Precision", "Recall", "F1", "Error"]
train_vals = [train_acc, train_p, train_r, train_f1, train_err]
val_vals = [val_acc, val_p, val_r, val_f1, val_err]
test_vals = [test_acc, test_p, test_r, test_f1, test_err]

print("\nBaseline Metrics Comparison:")
for name, tr, vl, ts in zip(metrics_names, train_vals, val_vals, test_vals):
    print(f"{name:<10} | Train: {tr:.4f} | Val: {vl:.4f} | Test: {ts:.4f}")

# --------------------------------------
# Print sample predictions (sanity check)
# --------------------------------------
print("\nSample Test Predictions (Baseline):")
for i in range(10):
    pred_label = "Positive" if test_preds[i] == 1 else "Negative"
    true_label = "Positive" if test_labels[i].item() == 1 else "Negative"
    print(f"Sample {i+1:02d} | Pred: {pred_label:<8} | True: {true_label}")

# --------------------------------------
# Plot metric comparison
# --------------------------------------
x = np.arange(len(metrics_names))
width = 0.25

plt.figure(figsize=(10,6))
plt.bar(x - width, train_vals, width, label="Train")
plt.bar(x, val_vals, width, label="Val")
plt.bar(x + width, test_vals, width, label="Test")
plt.xticks(x, metrics_names)
plt.ylabel("Metric Value")
plt.title("Baseline RoBERTa Metrics Comparison")
plt.legend()
plt.show()

# --------------------------------------
# Confusion Matrices
# --------------------------------------
for split, preds, labels in zip(
    ["Train", "Val", "Test"],
    [train_preds, val_preds, test_preds],
    [train_labels, val_labels, test_labels],
):
    cm = confusion_matrix(labels.numpy(), preds)
    disp = ConfusionMatrixDisplay(
        cm,
        display_labels=["Negative", "Positive"],
    )
    disp.plot(cmap="Blues")
    plt.title(f"{split} Confusion Matrix (Baseline)")
    plt.show()
