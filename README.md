Based on the provided Jupyter notebook and project title, here is a comprehensive summary and analysis of your project, **Hybrid Approach for Digital Fraud Detection in Online Financial Transactions**, using the **PaySim** dataset.

---

## ## Project Overview
This project develops a hybrid machine learning system to detect fraudulent transactions in financial data. By combining **Gradient Boosting** (XGBoost, LightGBM) with **Deep Learning** (LSTM), the model captures both tabular feature patterns and potential sequential dependencies in transaction behavior.

### ### Dataset Summary: PaySim
* **Total Transactions:** ~6.36 million.
* **Target Variable:** `isFraud` (1 for fraud, 0 for non-fraud).
* **Class Imbalance:** Highly imbalanced with only **8,213** fraud cases vs. ~6.35 million legitimate transactions.
* **Features Used:** `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, and `newbalanceDest`.

---

## ## Methodology & Pipeline

### 1. Data Preprocessing & Balancing
To handle the extreme class imbalance, you implemented **Under-sampling**:
* Extracted all 8,213 fraud cases.
* Sampled an equal number (8,213) of non-fraud cases.
* **Resulting Balanced Dataset:** 16,426 total transactions.
* **Feature Engineering:** Applied `LabelEncoder` for transaction types and performed a modulo operation on the `step` feature (`step % 24`) to simulate hourly cycles.

### 2. Model Training (Hybrid Components)
You trained three distinct types of models to compare and combine their strengths:

| Model | Type | Key Strength |
| :--- | :--- | :--- |
| **XGBoost** | Gradient Boosting | Excellent at handling tabular data and feature interactions. |
| **LightGBM** | Gradient Boosting | Faster training speed and high efficiency with large datasets. |
| **LSTM** | Deep Learning (RNN) | Captures temporal dependencies and patterns in sequences. |

### 3. The Hybrid "Stacking" Approach
In the final step, you implemented a **Meta-Classifier**:
* The predictions (probabilities) from the **XGBoost** model and the **LSTM** model were combined into a new feature set.
* A **Logistic Regression** model (Meta-learner) was trained on these combined predictions to make the final "fraud" or "non-fraud" decision.

---

## ## Key Findings & Performance

### ### Classification Results
All models, including the Hybrid Meta-model, achieved exceptional performance on the balanced test set:
* **Precision:** 0.99
* **Recall:** 0.99
* **F1-Score:** 0.99
* **Accuracy:** 99%

### ### Feature Importance (XGBoost vs. LightGBM)
* **XGBoost:** Identified **`newbalanceOrig`** as the most critical feature (~81.45% importance), suggesting that the change in the sender's account balance is the strongest indicator of fraud.
* **LightGBM:** Distributed importance more broadly, with **`amount`** and **`oldbalanceOrg`** being top contributors.


