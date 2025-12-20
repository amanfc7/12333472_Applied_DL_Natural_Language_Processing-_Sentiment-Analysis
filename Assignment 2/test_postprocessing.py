import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaModel
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

# CI-style Postprocessing Tests


DEVICE = torch.device("cpu")  # CPU for CI

TENSOR_DIR = "tokenized_files"
RESULTS_DIR = "train_val_results"
PLOTS_DIR = "train_val_plots"

class RobertaClassifier(nn.Module):
    """Minimal RoBERTa + linear head for postprocessing tests"""
    def __init__(self, dropout: float = 0.1, pooling: str = "cls"):
        super().__init__()
        self.pooling = pooling
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            pooled = (hidden * attention_mask.unsqueeze(-1)).sum(1)
            pooled = pooled / attention_mask.sum(1, keepdim=True)
        return self.classifier(self.dropout(pooled))


def load_tiny_subset(split="train", max_samples=8):
    inputs = torch.load(os.path.join(TENSOR_DIR, f"{split}_inputs.pt"))[:max_samples]
    masks = torch.load(os.path.join(TENSOR_DIR, f"{split}_masks.pt"))[:max_samples]
    labels = torch.load(os.path.join(TENSOR_DIR, f"{split}_labels.pt"))[:max_samples]
    return TensorDataset(inputs, masks, labels)


# Test 1: Tiny Training Pass

def test_forward_pass_cpu_tiny_training():
    """Run a tiny forward/backward pass on CPU"""
    dataset = load_tiny_subset()
    loader = DataLoader(dataset, batch_size=4)
    model = RobertaClassifier().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    model.train()
    for ids, masks, labels in loader:
        ids, masks, labels = ids.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(ids, masks)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        break  # just one batch for CI
    print("Tiny training forward/backward pass completed successfully.")



# Test 2: Logits Shape

def test_logits_shape():
    dataset = load_tiny_subset()
    loader = DataLoader(dataset, batch_size=4)
    model = RobertaClassifier().to(DEVICE)
    model.eval()

    for ids, masks, _ in loader:
        logits = model(ids.to(DEVICE), masks.to(DEVICE))
        assert logits.shape[1] == 2, f"Expected 2 classes, got {logits.shape[1]}"
        break
    print("Logits shape test passed.")


# Test 3: Predictions Validity

def test_predictions_validity():
    dataset = load_tiny_subset()
    loader = DataLoader(dataset, batch_size=4)
    model = RobertaClassifier().to(DEVICE)
    model.eval()

    for ids, masks, _ in loader:
        logits = model(ids.to(DEVICE), masks.to(DEVICE))
        preds = torch.argmax(logits, dim=1).cpu()
        assert set(preds.tolist()).issubset({0, 1})
        break
    print("Predictions validity test passed.")



# Test 4: Metrics Calculation

def test_metrics_computation():
    dataset = load_tiny_subset()
    loader = DataLoader(dataset, batch_size=4)
    model = RobertaClassifier().to(DEVICE)
    model.eval()

    all_preds, all_labels = [], []

    for ids, masks, labels in loader:
        logits = model(ids.to(DEVICE), masks.to(DEVICE))
        preds = torch.argmax(logits, dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        break

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0
    assert 0.0 <= f1 <= 1.0
    print(f"Metrics test passed. Acc={acc:.4f}, P={p:.4f}, R={r:.4f}, F1={f1:.4f}")


# Test 5: CSV Save/Load

def test_csv_save_load():
    df = pd.DataFrame({"preds": [0, 1, 1, 0], "labels": [0, 1, 0, 0]})
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_path = os.path.join(RESULTS_DIR, "test_postproc.csv")
    df.to_csv(file_path, index=False)
    assert os.path.exists(file_path)
    df_loaded = pd.read_csv(file_path)
    assert df_loaded.shape == df.shape
    print("CSV save/load test passed.")


# Test 6: Confusion Matrix Plotting

def test_confusion_matrix_plot():
    labels = [0, 1, 1, 0]
    preds = [0, 1, 0, 0]
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Neg", "Pos"])
    disp.plot()
    plt.close()
    print("Confusion matrix plot test passed.")
