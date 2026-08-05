# Credit Card Fraud Detection Web Application

A machine learning web application that predicts whether a credit card transaction is **Fraudulent** or **Legitimate** using a trained Logistic Regression model. The application provides real-time predictions through a Flask-based web interface.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-Learn
- Flask
- HTML/CSS
- Pickle

---

## Features

- Data preprocessing and cleaning
- Handling imbalanced transaction data
- Logistic Regression model training
- Model evaluation
- Model serialization using Pickle
- Real-time fraud prediction using Flask
- User-friendly web interface

---

## Workflow

1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Selection
5. Model Training
6. Model Evaluation
7. Save Trained Model
8. Flask-based Real-Time Prediction

---

## Problem Statement

Credit card fraud causes significant financial losses every year. This project uses machine learning to classify a transaction as **Fraudulent** or **Legitimate** based on transaction features.

---

## Model

**Algorithm Used**

- Logistic Regression

**Problem Type**

- Binary Classification

**Prediction Output**

- **0** → Legitimate Transaction
- **1** → Fraudulent Transaction

---

## Dataset

**Credit Card Fraud Detection Dataset (Kaggle)**

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Dataset Features

- **V1 – V28** : Anonymized transaction features generated using PCA.
- **Amount** : Transaction amount.
- **Class** : Target variable
  - 0 → Legitimate
  - 1 → Fraudulent

---

## Application Workflow

1. User enters transaction details through the web interface.
2. Flask receives the input.
3. Transaction data is preprocessed.
4. The trained Logistic Regression model predicts the probability of fraud.
5. The application displays whether the transaction is **Fraudulent** or **Legitimate**, along with a risk score.

---

## Future Improvements

- Hyperparameter tuning
- Additional machine learning models
- REST API endpoints
- Docker containerization
- Cloud deployment

---

## Author

**Karthik Siripurapu**
