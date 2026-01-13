
# Applied Deep Learning Project (WS2025)
**Assignment 3 – Deliver (IMDB Sentiment Analysis – Streamlit Web Applicationmodel)**  
**Name:** Aman Bhardwaj  
**Student ID:** 12333472  

---

# IMDB Sentiment Analysis – Streamlit Web Application

## Overview
This project is a web-based sentiment analysis application built using **Streamlit** and a **fine-tuned RoBERTa model**.  
The model was trained on the **IMDB movie reviews dataset** and predicts whether a given movie review expresses **positive** or **negative** sentiment.

This repository corresponds to **Assignment 3**, which focuses on **deploying the trained model** from Assignment 2 as an interactive web application.

---

## Model Description

- **Base Model:** RoBERTa-base (Hugging Face Transformers)
- **Architecture:**
  - RoBERTa encoder
  - Dropout layer
  - Linear classification head (2 classes: Positive, Negative)
- **Training Dataset:** IMDB Movie Reviews
- **Saved Model:** `best_model.pt` (from Assignment 2, loaded for inference)

The model used in this application is **not retrained**. Instead, the trained weights are loaded and used strictly for **inference** and **analysis**.

---

## Sentiment Prediction Pipeline

1. User enters a movie review in the web interface.
2. Text is tokenized using `RobertaTokenizer`.
3. The fine-tuned RoBERTa model produces raw logits.
4. **Temperature scaling** is applied to calibrate probabilities.
5. The final output includes:
   - Predicted sentiment (Positive / Negative)
   - Calibrated confidence score
   - Decision strength (logit margin)

---

## Decision Strength (Logit Margin)

Decision strength is defined as the absolute difference between the two output logits:

\[
\text{Decision Strength} = |z_{positive} - z_{negative}|
\]

This metric reflects the **internal certainty** of the model and is independent of probability calibration.

- Low values indicate ambiguous or neutral sentiment.
- High values indicate strong sentiment polarity.

This metric is used to flag:
- Borderline reviews
- Weak sentiment cases

---

## Temperature Scaling

Post-hoc temperature scaling is applied during inference:

softmax(logits / T)


### Temperature Scaling Details

Selected temperature: T = 2.5

Purpose:

Prevent overconfident predictions

Improve probability calibration

Preserve decision ranking

Extreme reviews (IMDB rating ≥ 7 or ≤ 4) typically yield confidence around 90–95%, while mid-range reviews produce lower confidence, reflecting natural ambiguity in sentiment expression.

## Web Application Features

Interactive text input for movie reviews

Real-time sentiment prediction

Confidence and decision strength display

Warning for neutral or borderline sentiment

Lightweight and CPU-friendly inference

## How to Run the Application

1.Navigate to Assignment 3 Directory

cd Assignment 3

2. Create and activate a virtual environment

python -m venv venv

### Windows
venv\Scripts\activate

### macOS/Linux
source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

4. Run Streamlit App

python -m streamlit run app.py

5. Open Browser: Visit: http://localhost:8501

## Important:

This app loads the trained model (best_model.pt) from Assignment 2 folder. Please make sure both Assignment 2 and Assignment 3 folders are in the same parent directory so the model path works correctly.

## Notes

- The IMDB dataset contains only binary sentiment labels.

- Neutral or mixed reviews are inherently ambiguous.


- The model is forced to choose a polarity, which is explicitly communicated via decision strength warnings.
