# HMDA 2023 — Home Loan Approval Prediction

Predicts home loan approval/denial using **real** U.S. mortgage application data from the
Home Mortgage Disclosure Act (HMDA), published annually by the CFPB.

---

## Why HMDA?

Unlike synthetic or anonymized datasets, HMDA data is:
- **Legally mandated** — lenders must report every application
- **Millions of rows** — 2023 dataset covers ~10M+ applications nationwide
- **Rich features** — income, DTI, LTV, race, sex, loan type, geography
- **Real outcomes** — actual approval/denial decisions

---

## Project Structure

```
hmda_loan_prediction/
├── data/
│   ├── raw/                    # Downloaded HMDA CSV
│   └── processed/              # Train/test splits (pkl)
├── models/                     # Trained model files (pkl)
├── outputs/
│   ├── eda/                    # EDA plots
│   └── evaluation/             # Evaluation plots + comparison CSV
├── scripts/
│   ├── 01_download_data.py     # Fetch data from CFPB API
│   ├── 02_eda.py               # Exploratory data analysis
│   ├── 03_preprocessing.py     # Clean, encode, split, SMOTE
│   ├── 04_train.py             # Train 3 models with K-fold CV
│   └── 05_evaluate.py          # Evaluate on test set, plots
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/your-username/hmda_loan_prediction
cd hmda_loan_prediction
pip install -r requirements.txt
```

---

## How to Run

Run scripts in order from the project root:

```bash
python scripts/01_download_data.py    # ~5-15 min depending on connection
python scripts/02_eda.py
python scripts/03_preprocessing.py
python scripts/04_train.py
python scripts/05_evaluate.py
```

---

## Target Variable

HMDA's `action_taken` field is binarized:

| action_taken | Meaning                  | Label    |
|--------------|--------------------------|----------|
| 1            | Loan originated          | Approved |
| 2            | Approved, not accepted   | Approved |
| 3            | Denied                   | Denied   |

Withdrawn (4), incomplete (5), and purchased (6) applications are excluded.

---

## Models

| Model               | Role                          |
|---------------------|-------------------------------|
| Logistic Regression | Interpretable baseline        |
| Random Forest       | Non-linear ensemble           |
| XGBoost             | Gradient boosting (often best)|

All models are compared using **5-fold stratified cross-validation** on the training set.
The best model (by ROC-AUC) is flagged for final evaluation on the held-out test set.

---

## Key Improvements Over Baseline

| Dimension          | Old (Kaggle)           | New (HMDA)                        |
|--------------------|------------------------|-----------------------------------|
| Data size          | 614 rows               | Millions of real applications     |
| Data source        | Synthetic/unknown      | Legally mandated real data        |
| Models             | Random Forest only     | LR + RF + XGBoost                 |
| Validation         | Single 80/20 split     | Stratified 5-fold CV              |
| SMOTE placement    | Potentially before split| Correctly applied after split    |
| Evaluation         | Accuracy + F1 only     | ROC-AUC, PR curves, threshold tuning |
| Feature count      | 12                     | 30+                               |

---

## Feature Engineering

- `income_to_loan_ratio` — applicant income relative to loan amount
- `loan_amount_log`, `applicant_income_log` — log transforms for skewed distributions
- DTI and LTV brackets converted to numeric midpoints
- Categorical variables one-hot encoded (loan type, property type, race, sex, etc.)

---

## Outputs

After running all scripts:

- `outputs/eda/` — 7 EDA plots (target distribution, loan amount, income, DTI, etc.)
- `outputs/evaluation/` — confusion matrices, ROC curves, PR curves, feature importance, threshold tuning
- `outputs/evaluation/model_comparison.csv` — final test set metrics for all models
- `models/optimal_threshold.pkl` — optimal decision threshold for the best model

---

## Data Source

**HMDA Data Browser** — Consumer Financial Protection Bureau  
https://ffiec.cfpb.gov/data-browser/

The dataset is public domain. No registration required.
