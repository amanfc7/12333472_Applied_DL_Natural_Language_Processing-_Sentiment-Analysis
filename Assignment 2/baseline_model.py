"""
Baseline evaluation using RoBERTa for IMDB sentiment classification.

This script:
- Uses the pretrained RoBERTa model with a default classification head
- Computes metrics, loss, and runtime on train, val, and test sets
- Saves all results to a CSV
- Prints metrics
- Plots metrics and loss curves
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.makedirs("baseline_model_results", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

BATCH_SIZE = 32  # baseline batch size (same as the batch size for the best optimized RoBerta model after training)
EPOCHS = 5       

# Baseline Model Definition:

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

# Metrics:

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    error = 1.0 - acc
    return acc, p, r, f1, error

# Load tokenized data:

def load_split(name: str):
    return (
        torch.load(f"tokenized_files/{name}_inputs.pt"),
        torch.load(f"tokenized_files/{name}_masks.pt"),
        torch.load(f"tokenized_files/{name}_labels.pt"),
    )

train_inputs, train_masks, train_labels = load_split("train")
val_inputs, val_masks, val_labels = load_split("val")
test_inputs, test_masks, test_labels = load_split("test")

# Initialize baseline model:

baseline_model = BaselineRobertaClassifier(pooling="cls").to(DEVICE)

for param in baseline_model.roberta.parameters():
    param.requires_grad = False

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    baseline_model.classifier.parameters(),
    lr=1e-3
)

train_loader = DataLoader(
    TensorDataset(train_inputs, train_masks, train_labels),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    TensorDataset(val_inputs, val_masks, val_labels),
    batch_size=BATCH_SIZE,
)

# training loop:

print("\nTraining baseline RoBERTa\n")

for epoch in range(1, EPOCHS + 1):
    baseline_model.train()
    total_loss = 0.0

    for ids, mask, lbl in train_loader:
        ids, mask, lbl = ids.to(DEVICE), mask.to(DEVICE), lbl.to(DEVICE)

        optimizer.zero_grad()
        logits = baseline_model(ids, mask)
        loss = loss_fn(logits, lbl)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * ids.size(0)

    avg_train_loss = total_loss / len(train_loader.dataset)

    baseline_model.eval()
    val_preds = []

    with torch.no_grad():
        for ids, mask, lbl in val_loader:
            ids, mask = ids.to(DEVICE), mask.to(DEVICE)
            logits = baseline_model(ids, mask)
            val_preds.extend(logits.argmax(1).cpu().numpy())

    val_acc, _, _, _, _ = compute_metrics(val_labels.numpy(), val_preds)

    print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f}")

# Evaluation function:

def evaluate_model(model, inputs, masks, labels, batch_size=BATCH_SIZE):
    loader = DataLoader(
        TensorDataset(inputs, masks, labels),
        batch_size=batch_size,
    )

    preds = []
    losses = []

    import time
    t_start = time.time()

    with torch.no_grad():
        for ids, mask, lbl in loader:
            ids, mask, lbl = ids.to(DEVICE), mask.to(DEVICE), lbl.to(DEVICE)
            logits = model(ids, mask)
            loss = loss_fn(logits, lbl)
            losses.extend([loss.item()] * ids.size(0))
            preds.extend(logits.argmax(1).cpu().numpy())

    runtime = time.time() - t_start
    acc, p, r, f1, err = compute_metrics(labels.numpy(), preds)
    avg_loss = np.mean(losses)
    return acc, p, r, f1, err, avg_loss, runtime, preds

# Evaluate on Train / Val / Test:

train_acc, train_p, train_r, train_f1, train_err, train_loss, train_runtime, train_preds = evaluate_model(
    baseline_model, train_inputs, train_masks, train_labels
)

val_acc, val_p, val_r, val_f1, val_err, val_loss, val_runtime, val_preds = evaluate_model(
    baseline_model, val_inputs, val_masks, val_labels
)

test_acc, test_p, test_r, test_f1, test_err, test_loss, test_runtime, test_preds = evaluate_model(
    baseline_model, test_inputs, test_masks, test_labels
)

# Print metrics:

print("\n BASELINE RoBERTa model METRICS ")
print(f"Train  - Acc: {train_acc:.4f}, Prec: {train_p:.4f}, Recall: {train_r:.4f}, F1: {train_f1:.4f}, Error: {train_err:.4f}, Loss: {train_loss:.4f}, Runtime: {train_runtime:.2f}s")
print(f"Val    - Acc: {val_acc:.4f}, Prec: {val_p:.4f}, Recall: {val_r:.4f}, F1: {val_f1:.4f}, Error: {val_err:.4f}, Loss: {val_loss:.4f}, Runtime: {val_runtime:.2f}s")
print(f"Test   - Acc: {test_acc:.4f}, Prec: {test_p:.4f}, Recall: {test_r:.4f}, F1: {test_f1:.4f}, Error: {test_err:.4f}, Loss: {test_loss:.4f}, Runtime: {test_runtime:.2f}s")
print("============================\n")

# Save metrics to CSV:

df_metrics = pd.DataFrame([
    {
        "split":"train", "accuracy":train_acc, "precision":train_p, "recall":train_r, "f1":train_f1,
        "error":train_err, "loss":train_loss, "runtime":train_runtime
    },
    {
        "split":"val", "accuracy":val_acc, "precision":val_p, "recall":val_r, "f1":val_f1,
        "error":val_err, "loss":val_loss, "runtime":val_runtime
    },
    {
        "split":"test", "accuracy":test_acc, "precision":test_p, "recall":test_r, "f1":test_f1,
        "error":test_err, "loss":test_loss, "runtime":test_runtime
    }
])
df_metrics.to_csv("baseline_model_results/baseline_metrics_comparison.csv", index=False)

# Plot metrics comparison:

metrics_list = ["accuracy","precision","recall","f1","error","loss"]
x = np.arange(len(metrics_list))
width = 0.25

plt.figure(figsize=(12,6))
plt.bar(x - width, df_metrics.iloc[0][metrics_list], width, label="Train")
plt.bar(x, df_metrics.iloc[1][metrics_list], width, label="Val")
plt.bar(x + width, df_metrics.iloc[2][metrics_list], width, label="Test")
plt.xticks(x, [m.upper() for m in metrics_list])
plt.ylabel("Value")
plt.title("Baseline RoBERTa Metrics Comparison")
plt.legend()
plt.savefig("baseline_model_results/baseline_metrics_comparison.png", bbox_inches="tight")
plt.show()

# Plot loss curve:

plt.figure()
plt.plot(["Train","Val","Test"], [train_loss, val_loss, test_loss], marker="o")
plt.title("Baseline RoBERTa Loss Comparison")
plt.ylabel("Cross-Entropy Loss")
plt.savefig("baseline_model_results/baseline_loss_curve.png", bbox_inches="tight")
plt.show()

# Confusion Matrices:

for split_name, preds, labels in zip(
    ["Train","Val","Test"],
    [train_preds, val_preds, test_preds],
    [train_labels, val_labels, test_labels],
):
    cm = confusion_matrix(labels.numpy(), preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Negative","Positive"])
    disp.plot(cmap="Blues")
    plt.title(f"{split_name} Confusion Matrix (Baseline)")
    plt.savefig(f"baseline_model_results/{split_name.lower()}_confusion_matrix.png", bbox_inches="tight")
    plt.close()  # close figure to avoid overlap

# ROC Curves for Train, Val, Test:

def plot_roc(labels, probs, split_name):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC={roc_auc:.4f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{split_name} ROC Curve (Baseline)")
    plt.legend(loc="lower right")
    plt.savefig(f"baseline_model_results/{split_name.lower()}_roc_curve.png", bbox_inches="tight")
    plt.close()

# Convert logits to probabilities for positive class:

def get_probs(model, inputs, masks):
    loader = DataLoader(TensorDataset(inputs, masks), batch_size=BATCH_SIZE)
    probs = []
    with torch.no_grad():
        for ids, mask in loader:
            ids, mask = ids.to(DEVICE), mask.to(DEVICE)
            logits = model(ids, mask)
            prob = torch.softmax(logits, dim=1)[:,1]
            probs.extend(prob.cpu().numpy())
    return np.array(probs)

# Plot and save ROC curves:

train_probs = get_probs(baseline_model, train_inputs, train_masks)
val_probs = get_probs(baseline_model, val_inputs, val_masks)
test_probs = get_probs(baseline_model, test_inputs, test_masks)

plot_roc(train_labels.numpy(), train_probs, "Train")
plot_roc(val_labels.numpy(), val_probs, "Val")
plot_roc(test_labels.numpy(), test_probs, "Test")
