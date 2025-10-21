# 12333472_Applied_DL_Natural_Language_Processing-_Sentiment-Analysis
Applied Deep Learning Project (WS 2025)

# Applied Deep Learning (WS 2025) – Assignment 1

**Name: Aman Bhardwaj**  
**Student ID: 12333472**  

---

## Project Overview

**Project Type:** Bring Your Own Method  
**Topic:** Natural Language Processing – Sentiment Analysis  
**Dataset:** IMDB Movie Reviews  

In the current era of massive online textual data, analyzing user-generated content, from reviews to social media posts, to automatically identify whether a text expresses a positive or negative sentiment has become critical for companies, researchers, and media analysts. I have selected **Natural Language Processing (NLP) for sentiment analysis** using the **IMDB Movie Reviews dataset**. This project will involve **text preprocessing, vectorization, model building, fine-tuning, and evaluation** using deep learning techniques.  

---

## References From Scientific Papers

1. **Kim Yoon – "Convolutional Neural Networks for Sentence Classification"**  
[Link](https://aclanthology.org/D14-1181.pdf)  

   **Summary:**  
   This paper demonstrated that Convolutional Neural Networks (CNNs), widely used in computer vision, can perform sentence-level classification tasks, including sentiment analysis. Using pre-trained word embeddings (Word2Vec) and convolutional filters on n-grams of words, the model extracts important features and applies max-pooling followed by a SoftMax classification layer.  

   **Key Points:**  
   - CNNs can automatically learn features from raw text.  
   - Achieved state-of-the-art results on multiple text classification benchmarks.  
   - Highlighted the importance of pre-trained embeddings for better performance.  

2. **Andrew L. Maas – "Learning Word Vectors for Sentiment Analysis"**  
[Link](https://aclanthology.org/P11-1015.pdf)  

   **Summary:**  
   Introduced the **IMDB Large Movie Review Dataset** (50,000 labeled reviews) and proposed using unsupervised word vectors combined with a simple neural network for sentiment classification. It showed that embeddings improve classification compared to bag-of-words approaches.  

   **Key Points:**  
   - First paper to introduce the IMDB dataset as a benchmark.  
   - Showed that embeddings capture semantic and sentiment information.  
   - Provides historical baselines for modern NLP techniques.  

3. **Jacob Devlin – "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"**  
[Link](https://arxiv.org/abs/1810.04805)  

   **Summary:**  
   Introduced **BERT**, a deep bidirectional transformer that learns contextual word representations using Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). It reads text both left-to-right and right-to-left, capturing rich context, and can be fine-tuned for NLP tasks like sentiment classification.  

   **Key Points:**  
   - Achieves state-of-the-art results on multiple NLP benchmarks.  
   - Pre-training on large corpora allows transfer learning to smaller datasets.  
   - Fine-tuning requires minimal task-specific changes but yields high performance.  

---

## Topic Decision

**Chosen Topic:** Natural Language Processing → Sentiment Analysis  

I chose sentiment analysis because it is widely used in industry for **customer feedback, market research, and social media monitoring**. It allows the use of **text preprocessing, embeddings, deep learning model training, fine-tuning, and evaluation**. Sentiment analysis automatically classifies text into positive, negative, or neutral categories, making it highly relevant for professional roles in **Data Science, Machine Learning, and AI Engineering**.  

---

## Decision on Project Type

**Chosen Project Type:** Bring Your Own Method  

This type focuses on **building or re-implementing a neural network architecture** on an existing, publicly available dataset. It emphasizes **model design, fine-tuning, and performance optimization**. The project allows experimentation with **preprocessing, tokenization, hyperparameters, and minor model modifications** to potentially improve performance. Selecting this type ensures the project emphasizes **state-of-the-art NLP modeling** and hands-on deep learning experience.  

---

## Project Summary and Approach

The project will build a **binary sentiment classifier** to predict whether a movie review is positive or negative.  

**Steps:**  
1. **Data Preprocessing**  
   - Clean text, remove special characters, lowercase text, remove stop-words.  
   - Split data into training, validation, and test sets.  
2. **Tokenization**  
   - Convert text into input IDs, attention masks, and segment IDs using a tokenizer.  
3. **Model Building**  
   - Fine-tune a pre-trained **BERT model**.  
   - Optionally, implement improvements to baseline methods.  
4. **Training & Hyperparameter Tuning**  
   - Experiment with learning rate, batch size, number of epochs.  
   - Implement early stopping to prevent overfitting.  
5. **Evaluation**  
   - Metrics: Accuracy, Precision, Recall, F1-score.  
   - Visualization: Confusion matrix, loss/accuracy plots.  
6. **Application Development**  
   - Build a small demo application to input custom reviews and predict sentiment.  

---

## Dataset Description

**Dataset:** IMDB Large Movie Review Dataset  
[Link](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  

**Details:**  
- 50,000 labeled reviews (25,000 train, 25,000 test).  
- Binary labels: Positive / Negative.  
- Balanced and clean dataset, suitable for deep learning NLP tasks.  
- Reviews cover a variety of genres and writing styles, providing a realistic benchmark.  

---

## Work Breakdown Structure & Time Estimates

| Step | Task | Estimated Hours |
|------|------|----------------|
| 1 | Initial Project planning, Scientific paper reviews, Dataset Collection | 4 |
| 2 | Dataset Exploration & Preprocessing | 5 |
| 3 | Tokenization & Data Pipeline Setup | 4 |
| 4 | Implementing the method and model development | 5 |
| 5 | Improving baseline or state-of-the-art methods | 5 |
| 6 | Training & Fine-Tuning | 10 |
| 7 | Evaluation & Visualization | 5 |
| 8 | Application Development | 5 |
| 9 | Report Writing | 5 |
| 10 | Presentation Preparation | 4 |
| **Total (approx.)** |  | **52** |

---

So, this project aims to produce a robust sentiment analysis model capable of classifying IMDB reviews with high accuracy. Expected outcomes include:
•	Performance metrics: Accuracy, Precision, Recall, F1-score.
•	A fine-tuned model adapted to the IMDB dataset.
•	Demonstration of predictions via a simple application and visualizations.
•	Insights on preprocessing, embedding strategies, and hyperparameter optimization.


---
