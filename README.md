# HMDA 2023 — Home Loan Approval Prediction

Predicts home loan approval/denial using **real** U.S. mortgage application data from the
Home Mortgage Disclosure Act (HMDA), published annually by the CFPB.

---

## Results

**Best model: XGBoost — ROC-AUC 0.9932 on 850,853 held-out applications**

### Cross-Validation (5-fold, training set)

| Model | Accuracy | F1 Macro | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 78.8% | 0.683 | 0.828 |
| Random Forest | 95.8% | 0.924 | 0.993 |
| **XGBoost** | **96.3%** | **0.928** | **0.993** |

### Test Set Performance (XGBoost)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Denied | 0.88 | 0.87 | 0.88 |
| Approved | 0.98 | 0.98 | 0.98 |
| **Overall accuracy** | | | **96.3%** |

### Top 10 Features (XGBoost)

| Feature | Importance |
|---|---|
| interest_rate | 0.600 |
| submission_of_application | 0.102 |
| debt_to_income_ratio | 0.076 |
| aus (automated underwriting system) | 0.035 |
| income_log | 0.013 |
| loan_term | 0.013 |
| loan_type_Conventional | 0.010 |
| equity_ratio | 0.009 |

### Key Findings

- **Interest rate dominates** at 0.60 importance — lenders price risk into the rate, so it encodes most of the creditworthiness signal.
- **DTI is the second strongest financial feature** at 0.076, consistent with standard underwriting practice.
- **Automated underwriting system matters** — applications processed outside standard AUS systems (Desktop Underwriter / Loan Prospector) behave very differently from those processed through them.
- **Logistic Regression is insufficient** — precision of only 0.39 on denied loans confirms the approval decision is a non-linear problem that tree-based models handle far better.
- **Optimal decision threshold is 0.50** — the default threshold is already optimal, no tuning required.

---

## Why HMDA?

Unlike synthetic or anonymized datasets, HMDA data is:
- **Legally mandated** — lenders must report every application by law
- **Massive scale** — 6.5M+ raw applications in the 2023 dataset
- **Rich features** — income, DTI, LTV, race, sex, loan type, AUS, geography
- **Real outcomes** — actual approval/denial decisions, not simulated

---

## Dataset

**Source:** HMDA Data Browser — Consumer Financial Protection Bureau  
**URL:** https://ffiec.cfpb.gov/data-browser/  
**Year:** 2023 | **Loan purpose:** Home purchase only | **Action taken:** 1, 2, 3

| Stat | Value |
|---|---|
| Raw rows | 6,554,532 |
| After filtering (actions 1/2/3 only) | 4,254,263 |
| Approved (label = 1) | 3,614,911 (85.0%) |
| Denied (label = 0) | 639,352 (15.0%) |
| Features after engineering | 121 |
| Training rows | 3,403,410 |
| Test rows | 850,853 |

The dataset is public domain. No registration required to download.

> **Note:** The raw CSV (~2.4 GB) is excluded from this repo. See download instructions below.

---

## Project Structure

```
hmda_loan_prediction/
├── data/
│   ├── raw/                    # Raw HMDA CSV (not in git — too large)
│   └── processed/              # Train/test pkl files (not in git)
├── models/                     # Trained model pkl files (not in git)
├── results/
│   ├── eda/                    # 13 EDA plots
│   └── evaluation/             # Evaluation plots + model_comparison.csv
├── scripts/
│   ├── eda.py               # 13-plot exploratory data analysis
│   ├── preprocessing.py     # Clean, encode, engineer, split
│   ├── train.py             # Train 3 models with 5-fold CV
│   └── evaluate.py          # Evaluate on test set, generate plots
├── .gitignore
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

## Data Download

1. Go to: https://ffiec.cfpb.gov/data-browser/data/2023?category=nationwide
2. Filter: **Loan Purpose = Home Purchase**, **Action Taken = 1, 2, 3**
3. Click **Download CSV**
4. Save to: `data/raw/year_2023_loan_purposes_1.csv`

---

## How to Run

Run scripts in order from the project root:

```bash
python scripts/02_eda.py               # Generate 13 EDA plots
python scripts/03_preprocessing.py     # Clean + engineer + split (~10 min)
python scripts/04_train.py             # Train all models with CV (~30-60 min)
python scripts/05_evaluate.py          # Evaluate + generate all plots
```

> `01_download_data.py` is included but the manual download above is more reliable for the full nationwide file.

---

## Target Variable

HMDA's `action_taken` field is binarized:

| action_taken | Meaning | Label |
|---|---|---|
| 1 | Loan originated | Approved (1) |
| 2 | Approved, not accepted | Approved (1) |
| 3 | Denied | Denied (0) |

Withdrawn, incomplete, and purchased applications are excluded.

---

## Feature Engineering

Starting from 99 raw HMDA columns, 26 were selected for modeling plus engineered features:

- `income_to_loan_ratio` — income divided by loan amount (affordability signal)
- `equity_ratio` — (property value − loan amount) / property value
- `loan_amount_log`, `income_log`, `property_value_log` — log transforms for right-skewed distributions
- DTI mixed encoding — plain integers kept as-is, bracket strings (`"30%-<36%"`) mapped to numeric midpoints
- Age brackets mapped to numeric midpoints (`"35-44"` → 39)
- 10 categorical columns one-hot encoded: state, race, sex, ethnicity, loan type, lien status, occupancy, credit score type, submission method, AUS

---

## Models

All three models use `class_weight="balanced"` to handle the 85/15 class imbalance without oversampling.

| Model | Role | Notes |
|---|---|---|
| Logistic Regression | Interpretable baseline | Confirms non-linearity of the problem |
| Random Forest | Non-linear ensemble | 300 trees, strong recall on denied loans |
| XGBoost | Gradient boosting | Best overall, strongest precision on denied |

Model selection uses **5-fold stratified cross-validation** on the training set, ranked by ROC-AUC.

---

## Outputs

| File | Description |
|---|---|
| `outputs/eda/01–13_*.png` | EDA plots (target, income, DTI, race, LTV, states, etc.) |
| `outputs/evaluation/01_confusion_matrices.png` | Side-by-side confusion matrices for all 3 models |
| `outputs/evaluation/02_roc_curves.png` | Overlaid ROC curves |
| `outputs/evaluation/03_pr_curves.png` | Precision-recall curves |
| `outputs/evaluation/04_feature_importance_*.png` | Top 20 features for RF and XGBoost |
| `outputs/evaluation/05_threshold_tuning.png` | Precision/recall/F1 vs decision threshold |
| `outputs/evaluation/model_comparison.csv` | Full test set metrics for all models |

---

## Comparison to Original Project

| Dimension | Original (Kaggle) | This Project (HMDA) |
|---|---|---|
| Data size | 614 rows | 4,254,263 real applications |
| Data source | Synthetic/unknown | Legally mandated real data |
| Models | Random Forest only | LR + Random Forest + XGBoost |
| Validation | Single 80/20 split | Stratified 5-fold CV |
| Class imbalance handling | SMOTE (possibly before split) | class_weight=balanced (correct) |
| Evaluation | Accuracy + F1 only | ROC-AUC, PR curves, threshold tuning, feature importance |
| Features | 12 | 121 (after encoding + engineering) |
| Best ROC-AUC | ~0.80 | **0.9932** |
