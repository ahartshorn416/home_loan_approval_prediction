# Home Loan Approval Prediction

## Project Overview
This project aims to **predict home loan approval** for applicants based on their personal, financial, and demographic details. The model is designed to assist financial institutions in **automating loan eligibility assessment** and identifying high-risk applicants.

The dataset contains **614 loan applications** with features such as income, credit history, education, and property area.

---

## Dataset
**Source:** Provided by the finance company.  

**Columns:**

| Column Name           | Description |
|----------------------|-------------|
| Loan_ID               | Unique loan identifier |
| Gender                | Applicant's gender (Male/Female) |
| Married               | Marital status (Yes/No) |
| Dependents            | Number of dependents (0,1,2,3+) |
| Education             | Education level (Graduate/Not Graduate) |
| Self_Employed         | Self-employment status (Yes/No) |
| ApplicantIncome       | Applicant's monthly income |
| CoapplicantIncome     | Coapplicant's monthly income |
| LoanAmount            | Loan amount requested |
| Loan_Amount_Term      | Loan term in months |
| Credit_History        | Credit history (1 = good, 0 = bad) |
| Property_Area         | Urban, Semiurban, Rural |
| Loan_Status           | Loan approval status (Y/N) |

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
- **Target Distribution:** Imbalance observed; more approved loans than rejected ones.
- **Categorical Analysis:** Credit history, education, marital status, and property area were analyzed against loan approval.
- **Numerical Analysis:** Applicant income, coapplicant income, and loan amount distributions analyzed. Skewed features log-transformed.

### 2. Data Preprocessing
- Missing values filled using **mode** (categorical) and **median** (numerical).  
- Categorical variables encoded into numeric format:
  - Binary mapping (0/1) for gender, education, married, self-employed, loan status.
  - One-hot encoding for property area.
- Feature engineering:
  - `TotalIncome = ApplicantIncome + CoapplicantIncome`
  - `LoanAmount_log = log(LoanAmount)`

### 3. Handling Class Imbalance
- The target variable is imbalanced (Approved > Not Approved).  
- Used **SMOTE (Synthetic Minority Oversampling Technique)** to oversample the minority class (rejected loans).

### 4. Modeling
- **Algorithm:** Random Forest Classifier  
- **Train-Test Split:** 80/20 stratified split  
- **Feature Scaling:** StandardScaler for numerical features  
- **Evaluation Metrics:** Precision, Recall, F1-score, Accuracy, ROC-AUC  

### 5. Model Performance (With SMOTE)
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Not Approved (0) | 0.69 | 0.66 | 0.68 | 38 |
| Approved (1) | 0.85 | 0.87 | 0.86 | 85 |

- **Accuracy:** 80%  
- **Macro F1-score:** 0.77  
- **Weighted F1-score:** 0.80  

**Insights:**  
- Model now predicts **rejected loans much better** (recall improved from ~0.42 → 0.66).  
- Approved loans remain highly accurate.  
- Key features driving approval: **Credit History, TotalIncome, LoanAmount, Property Area, Dependents**.

---

## How to Run

1. Clone the repository:
git clone https://github.com/ahartshorn416/home_loan_approval_prediction

2. Install dependencies:
pip install -r requirements.txt

3. Open the notebook:
jupyter notebook scripts/loan_approval_prediction.ipynb
