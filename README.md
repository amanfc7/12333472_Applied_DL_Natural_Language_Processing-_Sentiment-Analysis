# 12333472_Applied_DL_Natural_Language_Processing-_Sentiment-Analysis
Applied Deep Learning Project (WS 2025)

# Applied Deep Learning (WS 2025)  
**Assignment 1 – Initiate**  

**Name:**  
**Student ID:**  

---

For the project, I have selected the following as under:  
**Project Type:** Bring Your Own Method  
**Topic:** Natural Language Processing – Sentiment Analysis  
**Dataset:** IMDB Movie Reviews  

---

## Introduction

In the current era of massive online textual data, we have reviews to social media posts which automatically identify whether a text expresses a positive or negative sentiment. It has become a critical capability for companies, researchers, and media analysts. So, I selected the topic of Natural Language Processing (NLP) for the project of this Applied Deep Learning course. In this topic I will work on sentiment analysis. Here, I have selected the IMDB Movie Reviews dataset for the task of sentiment analysis using deep learning and NLP techniques. My project type will be Bring Your Own Method and it will require text preprocessing, vectorization, model building, fine-tuning, and evaluation.

---

## References From Scientific Papers

### 1) Kim Yoon – “Convolutional Neural Networks for Sentence Classification”  
**Link:** [https://aclanthology.org/D14-1181.pdf](https://aclanthology.org/D14-1181.pdf)  

This paper was among the first to demonstrate that Convolutional Neural Networks (CNNs), which are widely used in computer vision, can effectively perform sentence-level classification tasks, including sentiment analysis. This approach uses pre-trained word embeddings (Word2Vec) as input and applies convolutional filters of various sizes to extract features from n-grams of words. Max-pooling is applied to capture the most important features, followed by a fully connected SoftMax layer for classification.  

**Important points from this paper:**  
- It demonstrated that CNNs can learn features from raw text effectively.  
- It has achieved state-of-the-art results on multiple text classification benchmarks, including sentiment analysis.  
- It showed the importance of using pre-trained embeddings for better performance.  

So, this paper provides a foundational deep learning approach to sentiment analysis. The concept of learning text features automatically is very relevant for natural language processing techniques.  

### 2) Andrew L. Maas – “Learning Word Vectors for Sentiment Analysis”  
**Link:** [https://aclanthology.org/P11-1015.pdf](https://aclanthology.org/P11-1015.pdf)  

This paper introduced the IMDB Large Movie Review Dataset, which contains 50,000 labeled reviews. It proposed using unsupervised word vectors trained on a large corpus, combined with a simple neural network for sentiment classification. It has demonstrated that distributed word representations or embeddings improve sentiment classification performance over traditional bag-of-words approaches.  

**Important points from this paper:**  
- First paper to introduce the IMDB dataset as a benchmark for sentiment analysis.  
- Showed that word embeddings capture semantic and sentiment information.  
- Provides historical baselines and comparison points for modern deep learning approaches.  

This paper highlights the evolution from static embeddings (Word2Vec) to contextual embeddings (BERT) and supports the rationale for using pre-trained embeddings in modern sentiment analysis. This is very useful and highly related to the dataset I will use for the project.  

### 3) Jacob Devlin – “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”  
**Link:** [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)  

BERT (Bidirectional Encoder Representations from Transformers) introduced a deep bidirectional transformer that learns contextual word representations by pre-training on large corpora using Masked Language Modelling (MLM) and Next Sentence Prediction (NSP). It reads the text both left-to-right and right-to-left, capturing rich context. The model can then be fine-tuned on specific NLP tasks, like sentiment classification, by adding a simple classification layer.  

**Important points from this paper:**  
- Achieves state-of-the-art results on multiple NLP benchmarks.  
- Pre-training on large text corpora allows transfer learning to smaller datasets.  
- Fine-tuning requires minimal task-specific changes but yields high performance.  

---

## Topic Decision

**Chosen Topic:** Natural Language Processing → Sentiment Analysis  

I have selected this topic of NLP sentiment analysis as it is widely used in industry and research for customer feedback, market research, social media monitoring etc. and I am very interested in this topic. It enables using various techniques like text preprocessing, embeddings, deep learning model training, fine-tuning and evaluation. Sentiment analysis involves automatically classifying text data (such as reviews, tweets, or comments) into positive, negative or neutral sentiment categories.  

---

## Decision On Project Type

**Chosen Project:** Bring Your Own Method  

I have selected Bring Your Own Method for my project, which focuses on building or re-implementing a neural network architecture that operates on an existing, publicly available dataset. This emphasizes on model design, fine-tuning, and performance optimization. It will provide the opportunity to experiment with preprocessing, tokenization, hyperparameters and minor modifications to potentially improve performance. Overall, selecting this project type ensures that the focus remains on designing, implementing, and evaluating a deep learning model. It supports experimentation with state-of-the-art NLP techniques and to improve these techniques.  

---

## Project Summary and Approach

**Project Description:** The project will build a binary sentiment classifier that predicts whether a movie review is positive or negative using deep learning. It will involve the following steps:  

### 1) Data Preprocessing:  
- Clean text, remove special characters and other things.  
- Convert text to lowercase, remove stop-words etc.  
- Split data into training, validation, and test sets.  

### 2) Tokenization:  
- Using tokenizer to convert text into input IDs, attention masks, and segment IDs.  

### 3) Model Building:  
- Implementing my own idea and creating a model, improving the state of the art method.  
- Optimizing and Fine-tuning on dataset for binary sentiment classification.  

### 4) Training & Hyperparameter Tuning:  
- Experiment with learning rate, batch size, and number of epochs.  
- Implement early stopping to prevent overfitting.  

### 5) Evaluation:  
- Metrics: Accuracy, Precision, Recall, F1-score.  
- Visualization: Confusion matrix, loss and accuracy plots etc.  

### 6) Application Development:  
- A small application to run model and input custom reviews and get sentiment predictions, provide deliverables and to ship it.  

---

## Dataset Description

**Dataset:** IMDB Large Movie Review Dataset  
**Link:** [https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  

**Details:**  
- It has 50,000 labelled reviews: 25,000 train and 25,000 test.  
- It contains Binary labels: Positive and Negative.  
- It is a balanced dataset and is widely recognized and well-structured and clean, suitable for deep learning using Natural language processing.  

The reviews are primarily in English and cover a variety of genres, which helps in creating a model that generalizes well across different styles and topics. Each review contains free-form text with varying lengths, representing authentic user opinions and emotions, making it highly suitable for binary sentiment classification tasks. This combination of size, quality, and relevance makes the IMDB dataset an ideal choice for demonstrating sentiment analysis techniques and developing a robust NLP pipeline.  

---

## Work Breakdown Structure & Time Estimates

| Step | Task | Estimated Hours |
|------|------|----------------|
| 1 | Initial Project planning, Scientific paper reviews and Dataset Collection | 4 |
| 2 | Dataset Exploration & Preprocessing | 5 |
| 3 | Tokenization & Data Pipeline Setup | 4 |
| 4 | Implementing the method and model development | 5 |
| 5 | Improving baseline or state of the art methods | 5 |
| 6 | Training & Fine-Tuning | 10 |
| 7 | Evaluation & Visualization | 5 |
| 8 | Application Development | 5 |
| 9 | Report Writing | 5 |
| 10 | Presentation Preparation | 4 |
| **Total (approx.)** |  | **52** |

---

So, this project aims to produce a robust sentiment analysis model capable of classifying IMDB reviews with high accuracy. Expected outcomes include:  
- Performance metrics: Accuracy, Precision, Recall, F1-score.  
- A fine-tuned model adapted to the IMDB dataset.  
- Demonstration of predictions via a simple application and visualizations.  
- Insights on preprocessing, embedding strategies, and hyperparameter optimization.

