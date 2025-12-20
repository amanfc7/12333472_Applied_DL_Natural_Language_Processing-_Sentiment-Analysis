import pytest
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Define file  paths

RAW_DIR = Path("split_raw_data")
CLEAN_DIR = Path("preprocessed_clean_data")
TENSOR_DIR = Path("tokenized_files")

# Tests for Preprocessing Outputs

def test_csv_splits_exist():
    """Check that all CSV files exist"""
    for fname in ["train_raw.csv", "val_raw.csv", "test_raw.csv",
                  "train_clean.csv", "val_clean.csv", "test_clean.csv"]:
        file_path = RAW_DIR / fname if "raw" in fname else CLEAN_DIR / fname
        assert file_path.exists(), f"{file_path} does not exist"
    print("All CSV split files exist.")

def test_split_sizes():
    """Check that CSV splits have expected number of rows"""
    train_df = pd.read_csv(CLEAN_DIR / "train_clean.csv")
    val_df = pd.read_csv(CLEAN_DIR / "val_clean.csv")
    test_df = pd.read_csv(CLEAN_DIR / "test_clean.csv")
    total_rows = len(train_df) + len(val_df) + len(test_df)
    original_rows = 50000  # IMDB dataset
    assert total_rows == original_rows, f"Total rows {total_rows} != {original_rows}"
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

def test_cleaned_texts():
    """Verify cleaning worked (so no <br> tags, normalized spaces)"""
    sample_texts = pd.read_csv(CLEAN_DIR / "train_clean.csv")["review"].head(5)
    for text in sample_texts:
        assert "<br" not in text
        assert "  " not in text
    print("Sample cleaned texts checked successfully:", sample_texts.tolist())

def test_tokenized_tensors_exist():
    """to Check that tokenized tensors exist and shapes are correct"""
    for tensor_name in ["train_inputs.pt", "train_masks.pt", "train_labels.pt",
                        "val_inputs.pt", "val_masks.pt", "val_labels.pt",
                        "test_inputs.pt", "test_masks.pt", "test_labels.pt"]:
        file_path = TENSOR_DIR / tensor_name
        assert file_path.exists(), f"{file_path} missing"
        tensor = torch.load(file_path)
        print(f"{tensor_name} shape: {tensor.shape}")

def test_tensor_label_integrity():
    """Check that labels are only 0 or 1 and match number of samples"""
    for split in ["train", "val", "test"]:
        labels = torch.load(TENSOR_DIR / f"{split}_labels.pt")
        inputs = torch.load(TENSOR_DIR / f"{split}_inputs.pt")
        assert labels.shape[0] == inputs.shape[0], f"{split} labels/sample mismatch"
        assert set(torch.unique(labels).tolist()).issubset({0, 1}), f"{split} labels invalid"
    print("Labels integrity verified for all splits.")

def test_visual_sample_tokenization():
    """Print a few token IDs, attention masks, and labels for visual sanity check"""
    inputs = torch.load(TENSOR_DIR / "train_inputs.pt")
    masks = torch.load(TENSOR_DIR / "train_masks.pt")
    labels = torch.load(TENSOR_DIR / "train_labels.pt")

    print("Sample input IDs (first 2):\n", inputs[:2])
    print("Sample attention masks (first 2):\n", masks[:2])
    print("Sample labels (first 10):\n", labels[:10])


def test_review_length_distribution_plot():
    """to plot histogram of review lengths after cleaning"""
    train_df = pd.read_csv(CLEAN_DIR / "train_clean.csv")
    lengths = train_df["review"].apply(lambda x: len(x.split()))
    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=50, color='skyblue', edgecolor='black')
    plt.title("Distribution of Review Lengths (train_clean.csv)")
    plt.xlabel("Number of words")
    plt.ylabel("Number of reviews")
    plt.grid(True)
    plt.show()
    print("Review lengths plotted. Mean length:", lengths.mean(), "Max length:", lengths.max())
