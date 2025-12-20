# 12333472_Applied_DL_NLP_Sentiment_Analysis
# Applied Deep Learning Project (WS2025)
**Assignment 2 – Hacking (Sentiment Analysis on IMDB Movie Reviews Datast using RoBERTa model)**  
**Name:** Aman Bhardwaj  
**Student ID:** 12333472  

---

## Introduction

This assignment focuses on implementing a **binary sentiment classifier** for the IMDB Movie Reviews dataset which I selected for the project using deep learning. I am using RoBERTa model which uses a byte-level Byte-Pair Encoding (BPE) as its tokenization method. The RoBERTa model was pretrained on a massive amount of text data, including:

    Books
    News articles
    Web text
    Social media posts (specific variants were trained on datasets like 124 million tweets).
  
The goal of this assignment is to:

- Build an end-to-end pipeline: **preprocessing → tokenization → training → evaluation**.
- Implement a **baseline RoBERTa model**.
- Reimplement and Fine-tune the model to improve performance through **optimized hyperparameters**.
- Compare baseline and optimized models using standard metrics: Accuracy, Precision, Recall, F1-score, Loss, and Runtime. I amusing F1 as the **Main Metric** of comparison between the models.
- Generate visualizations for comparison.

**Key highlights:**

- Ran **8 different configurations** for the optimized model over ~15 hours.
- Determined the **best hyperparameters** for training/validation using Randomized Search (Grid Search for this task is too expensive).
- Evaluated optimized and baseline models on **train, validation, and test splits**.
- Saved metrics and plots for visual reporting.

---

## Preprocessing Pipeline

### Steps Implemented:

1. **Loading Dataset**
   - IMDB Large Movie Review Dataset: 50,000 labeled reviews.
   - Columns: `review` (text), `sentiment` (positive/negative).

2. **Train / Validation / Test Split**
   - Split **before preprocessing** to avoid leakage:
     - Train: 80%  
     - Validation: 10%  
     - Test: 10%  
   - Stratified split to maintain class balance.
   - Saved raw splits in `split_raw_data/`.

3. **Text Cleaning**
   - Removed **HTML tags** like `<br />`.
   - Normalized multiple spaces into a single space.
   - **No lowercasing, punctuation removal, stopword removal, or stemming**, as RoBERTa is pretrained on raw text.
   - Saved cleaned CSVs in `preprocessed_clean_data/`.

4. **Tokenization**
   - Used **RoBERTa tokenizer** to convert text into:
     - **Input IDs** (subword tokens)
     - **Attention masks**
   - Padded/truncated sequences to `max_length=256`.
   - Saved `.pt` files for train/val/test in `tokenized_files/`.

5. **Label Encoding**
   - Sentiment labels manually encoded as:
     - `0` → negative
     - `1` → positive
   - And then saved as tensors for training and evaluation.

---

## Preprocessing & Postprocessing Tests

Implemented **unit tests using `pytest`**:

1. **Preprocessing tests**
   - Verify CSV splits exist.
   - Check split sizes match original dataset.
   - Confirm cleaned texts have no `<br>` tags and normalized spaces.
   - Validate tokenized tensors exist and shapes are correct.
   - Check label integrity (only 0/1 and match input size).

2. **Postprocessing tests**
   - Tiny forward/backward pass with RoBERTa on CPU.
   - Check logits shape and predictions validity.
   - Metrics calculation (accuracy, precision, recall, F1).
   - Save/load CSV files and plot confusion matrix.

**Run tests with:**

```bash
python -m pytest -v

