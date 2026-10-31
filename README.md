# Python-Projects_Machine_Learning
Collection of machine learning projects in Python covering data preprocessing, exploratory analysis, model development, hyperparameter tuning, and evaluation.

1. Enhancing Job Matching with Machine Learning at Magnet (Nov 2024) — [`Thesis - Enhancing Job Matching/`](Thesis%20-%20Enhancing%20Job%20Matching/)

   This project improves personalized job recommendations using real-world Magnet.me interaction data (2022–2023). It addresses extreme data sparsity (~99.98%) by implementing a Bayesian Personalized Ranking (BPR) model with Matrix Factorization. The workflow included data preprocessing, hyperparameter tuning (Grid Search), and evaluation using ranking metrics (NDCG, Recall, Precision, MRR). Code versioning and experiment tracking were managed via GitHub, while model performance and tuning results were monitored and logged using NeptuneAI.

2. Data Analysis and Classification — [`Data Analysis and Classification/`](Data%20Analysis%20and%20Classification/)

   **Cafe Sales Analysis** (`cafe_sales_manipulation.ipynb`): cleans a messy cafe transactions dataset (UNKNOWN/ERROR values, invalid numbers, missing dates), recovers missing totals from quantity × price, and analyses revenue by item, location, payment method and month. Also covers spend bands, groupby/agg summaries, pivot tables, MultiIndex selection, merging item costs to compute profit per category, and wide/long reshaping with `pivot_table` and `melt`.

   **Credit Card Fraud Detection** (`fraud_detection_logistic_regression.ipynb`): binary classification of 10,000 card transactions (amount, hour, merchant category, foreign/location-mismatch flags, device trust score, 24h velocity, cardholder age), where only ~1.5% are fraudulent. The notebook covers EDA (distributions, percentiles, IQR outlier detection, fraud rate by segment), one-hot encoding and scaling, and a Logistic Regression inside an imbalanced-learn pipeline with SMOTE, tuned with 5-fold GridSearchCV on F1. On the held-out test set it catches 87% of fraud cases (recall 0.87, precision 0.39, F1 0.53), with per-fold train/validation gaps checked for overfitting.
