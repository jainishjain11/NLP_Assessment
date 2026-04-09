# Aspect-Based Sentiment Analysis (ABSA) on Restaurant Reviews

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jainishjain11/NLP_Assessment/blob/main/NLP_Assessment_updated.ipynb)

## 📌 Overview
This repository contains a complete Natural Language Processing (NLP) pipeline for performing **Aspect-Based Sentiment Analysis (ABSA)** on restaurant reviews. Rather than analyzing the overall sentiment of a review, this project identifies specific aspects of the dining experience and determines the sentiment associated with each. 

The analysis focuses on four key aspects:
* 🍔 **Food**
* 💁 **Service**
* 🍷 **Ambience**
* 💵 **Price**

## 📊 Dataset
The project uses the `Restaurant_Reviews.tsv` dataset. The raw data is unzipped and loaded directly into a pandas DataFrame for processing.

## ⚙️ Methodology & Pipeline

### 1. Text Preprocessing
Before model training, all raw reviews undergo a strict cleaning process:
* Lowercasing all text.
* Removing non-alphabetic characters (punctuation and numbers).
* Removing English stop words using `nltk`.
* **Lemmatization** via `WordNetLemmatizer` to reduce words to their base forms (e.g., "tasty" to "taste").

### 2. Aspect Extraction (Keyword-Based)
Aspects are identified using a predefined dictionary of keywords. For example:
* **Food:** *food, taste, delicious, burger, steak, pizza, salad, menu, flavor, meat, fresh*
* **Service:** *service, waiter, staff, server, friendly, attentive, slow, rude, hospitality*
* **Ambience:** *ambience, atmosphere, decor, music, lighting, vibe, place, setting, view*
* **Price:** *price, cost, expensive, cheap, value, bill, affordable, overpriced, money*

### 3. Feature Engineering
* Features are extracted using an optimized **TF-IDF Vectorizer** (Term Frequency-Inverse Document Frequency).
* N-gram range: (1, 3)
* Max features: 3000

### 4. Models Evaluated
For each of the four identified aspects, the reviews are split into training (80%) and testing (20%) sets. Three different models are trained and evaluated:
1. **Rule-Based Baseline:** A custom approach comparing counts of predefined positive and negative words.
2. **Logistic Regression:** A supervised learning algorithm utilizing balanced class weights.
3. **Linear Support Vector Classifier (LinearSVC):** A powerful classification algorithm for text analysis.

## 📈 Results & Evaluation
The models are evaluated using **Accuracy** and **Macro F1-Score**. The output is dynamically generated into a structured summary table for easy comparison across models and aspects. Supervised models (LogReg and SVM) consistently outperform the rule-based baseline.

## 🚀 How to Run the Code

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/jainishjain11/NLP_Assessment.git](https://github.com/jainishjain11/NLP_Assessment.git)
   cd NLP_Assessment
   ```
2. **Install the required dependencies:**
   Make sure you have Python 3 installed. You will need the following libraries:
   ```bash
   pip install pandas numpy nltk scikit-learn
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook NLP_Assessment_updated.ipynb
   ```
   Alternatively, you can just click the "Open in Colab" badge at the top of this README to run the code directly in your browser without local setup!

🛠 Libraries Used
pandas, numpy (Data manipulation)
nltk (Natural Language Toolkit for preprocessing)
scikit-learn (Machine learning models and TF-IDF)
re (Regular expressions)
