import os
import re
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer


class IMDBPreprocessing:
    """
    Preprocessing pipeline for the IMDB sentiment analysis dataset.

    This class performs:
    1. Loading the raw IMDB dataset
    2. Splitting the dataset into Train / Validation / Test (80/10/10)
       before any preprocessing to avoid data leakage
    3. Minimal text preprocessing as I will use RoBERTa (which is pretrained on raw text)
    4. Saving raw splits and cleaned splits as CSV files
    5. Tokenization using RoBERTa tokenizer (BPE-based)
    6. Saving tokenized tensors as .pt files for further training
    """

    def __init__(
        self,
        data_path="IMDB Dataset.csv",
        max_length=256,
        test_size=0.10,
        val_size=0.10,
        random_state=42,
    ):
        """
        It initializes preprocessing parameters and tokenizer.

        Args:
            data_path (str): Path to the raw IMDB CSV file
            max_length (int): Maximum sequence length for RoBERTa
            test_size (float): Fraction of dataset used for test split
            val_size (float): Fraction of remaining data used for validation
            random_state (int): Seed for reproducibility
        """

        self.data_path = data_path
        self.max_length = max_length
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        # to initialize RoBERTa tokenizer (uses Byte-Pair Encoding internally):

        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        # Create output directories if they do not exist
        
        os.makedirs("split_raw_data", exist_ok=True)
        os.makedirs("preprocessed_clean_data", exist_ok=True)
        os.makedirs("tokenized_files", exist_ok=True)
    
    # text preprocessing (minimal and Transformer-friendly)
    
    def clean_text(self, text):
        """
        Applies minimal preprocessing to raw text.

        Importantly, not done here is:
        - No lowercasing
        - No stopword removal
        - No stemming / lemmatization
        - No manual tokenization

        It is because RoBERTa is pretrained on raw text.

        Steps applied here are:
        - Remove HTML tags
        - Normalize whitespace

        Args:
            text (str): Raw review text

        It returns:
            str: Cleaned text
        """

        # to remove HTML tags such as <br />
        text = re.sub(r"<.*?>", " ", text)

        # to normalize multiple whitespaces into a single space
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    # Tokenization:
    
    def tokenize_texts(self, texts):
        """
        Tokenizes a list of texts using RoBERTa tokenizer.

        Args:
            texts (pd.Series): Series of cleaned text samples

        Returns:
            input_ids (torch.Tensor): Token IDs
            attention_mask (torch.Tensor): Attention masks
        """

        encodings = self.tokenizer(
            texts.tolist(),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return encodings["input_ids"], encodings["attention_mask"]

   
    # Full preprocessing pipeline
    
    def run(self):
        """
        Executes the full preprocessing pipeline:
        - Load raw dataset
        - Split into train/val/test (80/10/10)
        - Save raw splits
        - Apply minimal preprocessing
        - Save cleaned CSVs
        - Tokenize text
        - Save tokenized tensors
        """

        print("Loading raw dataset (IMDB dataset)...")
        df = pd.read_csv(self.data_path)
        print("Raw dataset shape:", df.shape)

        
        # 1: Split before preprocessing:
    
        print("Splitting dataset (80/10/10)...")

        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df["sentiment"],
        )

        train_df, val_df = train_test_split(
            train_df,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=train_df["sentiment"],
        )

        print("Train size:", train_df.shape)
        print("Validation size:", val_df.shape)
        print("Test size:", test_df.shape)

        # Save RAW splits (before preprocessing):

        train_df.to_csv("split_raw_data/train_raw.csv", index=False)
        val_df.to_csv("split_raw_data/val_raw.csv", index=False)
        test_df.to_csv("split_raw_data/test_raw.csv", index=False)

        
        # 2: Minimal preprocessing:
        
        print("Applying minimal text preprocessing...")
        
        train_df["clean_review"] = train_df["review"].apply(self.clean_text)
        val_df["clean_review"] = val_df["review"].apply(self.clean_text)
        test_df["clean_review"] = test_df["review"].apply(self.clean_text)

        # Save CLEANED splits (after preprocessing):

        train_df.to_csv("preprocessed_clean_data/train_clean.csv", index=False)
        val_df.to_csv("preprocessed_clean_data/val_clean.csv", index=False)
        test_df.to_csv("preprocessed_clean_data/test_clean.csv", index=False)

        # 3: Tokenization
        
        print("Tokenizing text using RoBERTa tokenizer...")

        train_inputs, train_masks = self.tokenize_texts(train_df["clean_review"])
        val_inputs, val_masks = self.tokenize_texts(val_df["clean_review"])
        test_inputs, test_masks = self.tokenize_texts(test_df["clean_review"])

        # to convert sentiment labels to integers:

        label_map = {"negative": 0, "positive": 1}

        train_labels = torch.tensor(train_df["sentiment"].map(label_map).values)
        val_labels = torch.tensor(val_df["sentiment"].map(label_map).values)
        test_labels = torch.tensor(test_df["sentiment"].map(label_map).values)

        # 4: Save tokenized tensors
      
        print("Saving tokenized tensors...")

        torch.save(train_inputs, "tokenized_files/train_inputs.pt")
        torch.save(train_masks, "tokenized_files/train_masks.pt")
        torch.save(train_labels, "tokenized_files/train_labels.pt")

        torch.save(val_inputs, "tokenized_files/val_inputs.pt")
        torch.save(val_masks, "tokenized_files/val_masks.pt")
        torch.save(val_labels, "tokenized_files/val_labels.pt")

        torch.save(test_inputs, "tokenized_files/test_inputs.pt")
        torch.save(test_masks, "tokenized_files/test_masks.pt")
        torch.save(test_labels, "tokenized_files/test_labels.pt")

        print("Preprocessing completed. Data is now ready for training :)")
