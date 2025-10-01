💻 Fraud Prediction Analyzer

Fraud Prediction Analyzer is an interactive web application that predicts whether a financial transaction is likely to be fraudulent. It delivers real-time results with a probability score, helping users instantly assess transaction risk.

🚀 Features

Interactive transaction input form

Real-time prediction: Fraud / Not Fraud

Probability/confidence score for risk assessment

Color-coded high/low risk indicators

Mobile-friendly and responsive UI

🔧 How It Works

Users provide key transaction details like:

Transaction amount

Originator balance ratio

Transaction type

Time of transaction

Binary risk flags/indicators

The backend model analyzes these inputs and returns:

✅ Fraud status

✅ Probability score

🎯 Purpose

Designed for:

Financial institutions

Fraud analysts

Security teams

Individual users handling transactions

It enables fast detection of suspicious activity and supports proactive fraud prevention.

🛠️ Technology Stack

Frontend: HTML, Tailwind CSS, JavaScript
Backend: Python (Flask)
Model: Pre-trained Machine Learning Model for Fraud Detection

🧠 About the Model

The fraud detection model is designed for highly imbalanced transaction data:

Logistic Regression is used for baseline evaluation and probability calibration.

XGBoost is used for high-performance classification, capturing complex patterns in the data.

Imbalance Handling: SMOTE (oversampling) and RandomUnderSampler are used to address class imbalance.

Evaluation Metrics: Precision-Recall Curve, ROC-AUC, confusion matrix, and classification report.

Explainability: SHAP values are used to interpret model predictions and feature importance.

Workflow:

Preprocess transaction data and handle missing values.

Apply resampling techniques (SMOTE/undersampling) for balanced training.

Train the model (Logistic Regression / XGBoost).

Predict Fraud/Not Fraud and return probability score.

Use SHAP to visualize feature impact on predictions.
