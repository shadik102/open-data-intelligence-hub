# Reusable ML Pipeline with scikit-learn

## Project Overview

This project demonstrates a reusable Machine Learning Pipeline using scikit-learn for Customer Churn Prediction.

The pipeline performs:

* Data Loading
* Train-Test Split
* Numerical Data Preprocessing
* Categorical Data Preprocessing
* Feature Transformation using ColumnTransformer
* Model Training using RandomForestClassifier
* Prediction and Evaluation
* Pipeline Saving using Joblib

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib

## Files

* churn.csv
* main.py
* churn_pipeline.pkl
* requirements.txt

## Model

RandomForestClassifier

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score

## Output

The trained pipeline is saved as:

churn_pipeline.pkl

This saved pipeline can be loaded later and used for predictions on new customer data.
