# E-Commerce Fraud Detection

A machine learning pipeline that predicts whether an online card transaction is fraudulent.

## Dataset

**[E-Commerce Fraud Detection Dataset](https://www.kaggle.com/datasets/umuttuygurr/e-commerce-fraud-detection-dataset)** by UmutUygurr on Kaggle. License: CC0 Public Domain.

> **Note:** This is **synthetic data**. It was generated to mimic real fraud patterns, so it contains no real customer or card information. Model scores on it will likely be higher than on real-world data.

| | |
|---|---|
| File | `transactions.csv` (25 MB) |
| Rows | 299,695 transactions |
| Columns | 17 |
| Users | 6,000 (40–60 transactions each) |
| Period | 1 Jan 2024 – 31 Oct 2024 |
| Target | `is_fraud` |
| Fraud rate | 2.21% (6,612 fraudulent transactions) |
| Missing values | None |

### Columns

| Column | Type | Description |
|---|---|---|
| `transaction_id` | int | Unique transaction ID |
| `user_id` | int | Customer ID (1–6,000) |
| `account_age_days` | int | Age of the user's account in days (1–1,890). Same value on every row of a user |
| `total_transactions_user` | int | Number of transactions the user has in the dataset (40–60). Same value on every row of a user |
| `avg_amount_user` | float | The user's typical spend per transaction. Same value on every row of a user |
| `amount` | float | Transaction amount (1.00–16,994.74; currency not specified) |
| `country` | category | The user's country: `DE`, `ES`, `FR`, `GB`, `IT`, `NL`, `PL`, `RO`, `TR`, `US` |
| `bin_country` | category | Country of the bank that issued the card, based on the card's BIN (first digits of the card number). Same 10 codes |
| `channel` | category | Where the purchase was made: `web` or `app` |
| `merchant_category` | category | `electronics`, `fashion`, `gaming`, `grocery`, `travel` |
| `promo_used` | 0/1 | 1 = a promo code was used |
| `avs_match` | 0/1 | Address Verification System: 1 = the billing address matches the card issuer's records |
| `cvv_result` | 0/1 | 1 = the card security code (CVV) check passed |
| `three_ds_flag` | 0/1 | 1 = 3-D Secure authentication (e.g. bank app or SMS confirmation) was used |
| `transaction_time` | datetime (UTC) | Date and time of the transaction |
| `shipping_distance_km` | float | Shipping distance in km (0–3,749). The dataset does not state the reference point |
| `is_fraud` | 0/1 | **Target:** 1 = fraudulent transaction |

### Fraud rate by key signal

| Signal | Fraud rate when present | Fraud rate otherwise |
|---|---|---|
| Card country ≠ user country | 11.3% | 1.4% |
| CVV check failed | 10.6% | 1.0% |
| Address check (AVS) failed | 9.7% | 0.8% |
| No 3-D Secure | 6.8% | 1.0% |
| Promo code used | 4.5% | 1.8% |
| Web purchase (vs. app) | 3.6% | 0.8% |

### Things to know

- **Imbalanced classes:** only about 2% of transactions are fraud, so accuracy is misleading (predicting "not fraud" for everything scores 97.8%). Models are compared with **PR-AUC**, plus precision and recall at the chosen threshold.
- **User-level columns:** `account_age_days`, `total_transactions_user`, `avg_amount_user` and `country` are fixed per user. `total_transactions_user` counts the user's transactions across the whole 10 months, so it includes information from the future when the data is split by time.

## Getting the data

The dataset is **not included in this repository**: the `data/` folder is listed in `.gitignore`. To run the project:

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/umuttuygurr/e-commerce-fraud-detection-dataset), or with the [Kaggle CLI](https://github.com/Kaggle/kaggle-api):
   ```bash
   kaggle datasets download -d umuttuygurr/e-commerce-fraud-detection-dataset -p data/raw --unzip
   ```
2. Make sure the file is at `data/raw/transactions.csv`.

## Project structure

```
├── data/                 # not tracked by git
│   ├── raw/              # original transactions.csv
│   └── processed/        # cleaned data
├── notebook-eda/
│   └── eda.ipynb         # exploratory data analysis
├── src/
│   ├── preprocessing.py  # loading, cleaning, train/test split
│   ├── features.py       # feature engineering
│   ├── models.py         # model training
│   └── evaluate.py       # metrics and evaluation
└── main/
    └── main.py           # runs the full pipeline
```
