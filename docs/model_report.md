# Campaign Conversion Prediction Model Report

## 1. Overview
This report details the XGBoost-based Machine Learning model developed to predict the probability of a user converting from a marketing campaign contact into a customer. This forms the ML service of our dual-service Customer Intelligence Platform.

## 2. Dataset & Feature Engineering
The model was trained on the `campaign.csv` dataset, which contains demographic, behavioral, and campaign-specific interactions.
- **Numeric Features (7):** `age`, `income`, `num_campaigns`, `num_clicks`, `num_opens`, `recency_days`, `tenure_days`. (Missing values imputed with 0).
- **Categorical Features (4):** `gender`, `channel`, `product_category`, `region`. (Encoded using LabelEncoder; unseen categories during inference are safely encoded to `-1`).
- **Target Variable:** `converted` (Binary: 0 or 1).

## 3. Model Architecture
- **Algorithm:** XGBoost Classifier (`xgboost.XGBClassifier`)
- **Hyperparameters:**
  - `n_estimators`: 100
  - `learning_rate`: 0.1
  - `max_depth`: 5
  - `eval_metric`: 'logloss'
  - `early_stopping_rounds`: 10
- **Validation:** 20% Stratified Test Split

## 4. Performance Metrics
The model achieved highly robust performance metrics on the test dataset:
- **ROC-AUC Score:** 0.9069
- **F1 Score:** 0.9924

## 5. Artifact Delivery
The trained model, including its numeric feature schemas, categorical columns, and fitted label encoders, are packaged together via `pickle` into a unified `conversion_model.pkl` artifact. This ensures that the FastAPI inference service can exactly replicate the feature engineering transformations that occurred during training.
