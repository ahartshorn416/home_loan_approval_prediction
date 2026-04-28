""""
preprocessing.py

Author: Alison Hartshorn
Project: Home Loan Approval Prediction

Cleans,encodes, and engineers features from the HMDA 2023 dataset.
Tailored to the actual column types in year_2023_loan_purposes_1.csv:

- loan_to_value_ratio: numeric float(NaN=exempt)
- interest_rate: numeric float(NaN=exempt)
- debt_to_income_ratio: MIXED — integers AND bracket strings AND NaN
- applicant_age: string brackets("<25", "25-34", etc.)

Saves train / test splits + scaler + feature names to data processed.
"""
# ------------
# Imports
# -------------
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# ------------
# Paths
# -------------
DATA_FILE = Path(r"C:\Users\alica\Downloads\year_2023_loan_purposes_1.csv")
PROC_DIR  = Path("C:\\Users\\alica\\OneDrive\\Documents\\home_loan_approval_prediction\\data\\processed_data")
PROC_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.20

# --------------
# Target mapping
# --------------
# 1=Originated, 2=Approved not accepted → Approved (1)
# 3=Denied → Denied (0)
ACTION_MAP = {1: 1, 2: 1, 3: 0}

# ---------------
# Columns to load
# ---------------
# Chosen for predictive value; excludes identifiers and post-decision leakage
USECOLS = [
    "action_taken",
    # Loan characteristics
    "loan_amount", "loan_type", "loan_term", "lien_status",
    "loan_to_value_ratio", "interest_rate", "conforming_loan_limit",
    "occupancy_type", "property_value",
    # Applicant financials
    "income", "debt_to_income_ratio",
    # Applicant demographics (HMDA-reported)
    "derived_race", "derived_sex", "derived_ethnicity",
    "applicant_age", "applicant_age_above_62",
    "co-applicant_age_above_62",
    # Application process
    "applicant_credit_score_type", "submission_of_application",
    "aus-1",
    # Geography / census tract
    "state_code",
    "tract_minority_population_percent",
    "ffiec_msa_md_median_family_income",
    "tract_to_msa_income_percentage",
    "tract_owner_occupied_units",
    "tract_median_age_of_housing_units",
]

# -----------------------------------------
# DTI: mixed string/int to numeric midpoint
# -----------------------------------------
# Plain integers (36–49) are already numeric and used as-is.
# Bracket strings are mapped to midpoints.
DTI_BRACKET_MAP = {
    "<20%":     15.0,
    "20%-<30%": 25.0,
    "30%-<36%": 33.0,
    "50%-60%":  55.0,
    ">60%":     65.0,
    "Exempt":   np.nan,
    "NA":       np.nan,
}

# -------------------------------
# Age bracket to numeric midpoint
# -------------------------------
AGE_MAP = {
    "<25":  22.0,
    "25-34": 29.0,
    "35-44": 39.0,
    "45-54": 49.0,
    "55-64": 59.0,
    "65-74": 69.0,
    ">74":   78.0,
    "8888": np.nan,  # HMDA code for not provided
}

# ------------
# Loan type
# -------------
LOAN_TYPE_MAP = {1:"Conventional", 2:"FHA", 3:"VA", 4:"RHS"}

# ------------
# Occupancy type
# -------------
OCCUPANCY_MAP = {1:"Principal", 2:"SecondHome", 3:"Investment"}

# ------------
# Lien status
# -------------
LIEN_MAP = {1:"FirstLien", 2:"SubordinateLien"}

# --------------------
# Conforming loan limit
# ---------------------
# C=conforming, NC=non-conforming, U=undetermined, NA=not applicable
CONFORMING_MAP = {"C": 1, "NC": 0, "U": np.nan, "NA": np.nan}

# -----------------------------
# Automated underwriting system
# ------------------------------
# 1=Desktop Underwriter, 2=Loan Prospector, 3=Technology Open..., 4=Guaranteed...,
# 5=TOTAL, 6=Other, 7=Not applicable, 8=Not provided, 1111=Exempt
AUS_MAP = {
    1: "DesktopUnderwriter", 2: "LoanProspector", 3: "TechOpen",
    4: "Guaranteed", 5: "TOTAL", 6: "Other"
}

# -------------------------
# Submission of application
# -------------------------
# 1=Submitted directly, 2=Not submitted directly, 3=Not applicable, 1111=Exempt
SUBMISSION_MAP = {1: "Direct", 2: "NotDirect"}

# -----------------
# Credit score type
# ------------------
# 1=Equifax, 2=Experian, 3=FICO, 4=FICO9, 5=VantageScore3, 6=VantageScore4,
# 7=More than one, 8=Other, 9=Not applicable, 1111=Exempt
# Simplify: grouped vs not
CREDIT_SCORE_MAP = {
    1:"Equifax", 2:"Experian", 3:"FICO Classic", 4:"FICO9",
    5:"VantageScore3", 6:"VantageScore4", 7:"Multiple", 8:"Other"
}


def load_data() -> pd.DataFrame:
    print(f"[INFO] Loading data from {DATA_FILE} ...")
    df = pd.read_csv(DATA_FILE, usecols=USECOLS, low_memory=False)
    print(f"       Raw shape: {df.shape}")
    return df


def map_target(df: pd.DataFrame) -> pd.DataFrame:
    df["label"] = df["action_taken"].map(ACTION_MAP)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    df.drop(columns=["action_taken"], inplace=True)
    approved = df["label"].sum()
    denied   = len(df) - approved
    print(f"       Approved: {approved:,} ({approved/len(df)*100:.1f}%) | "
          f"Denied: {denied:,} ({denied/len(df)*100:.1f}%)")
    return df


def encode_dti(df: pd.DataFrame) -> pd.DataFrame:
    """
    DTI is mixed: plain integers like 36, 37,...49 and bracket strings.
    Strategy: convert bracket strings to midpoints, keep integers as - is.

    """
    col = "debt_to_income_ratio"
    dti = df[col].astype(str).str.strip()

    # Try numeric first
    numeric = pd.to_numeric(dti, errors="coerce")

    # For non-numeric rows, map brackets
    bracket = dti.map(DTI_BRACKET_MAP)

    # Combine: use numeric where available, else bracket midpoint
    df[col] = numeric.combine_first(bracket)
    return df


def encode_age(df: pd.DataFrame) -> pd.DataFrame:
    col = "applicant_age"
    df[col] = df[col].astype(str).str.strip().map(AGE_MAP)
    return df


def encode_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Yes / No and 1 / 2 binary columns to 0 / 1.
    """
    # applicant_age_above_62: "Yes", "No", "NA"
    for col in ["applicant_age_above_62", "co-applicant_age_above_62"]:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).astype(float)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map coded columns and one - hot encode string columns.
    """

    # Numeric code mappings
    if "loan_type" in df.columns:
        df["loan_type"] = df["loan_type"].map(LOAN_TYPE_MAP).fillna("Other")

    if "occupancy_type" in df.columns:
        df["occupancy_type"] = df["occupancy_type"].map(OCCUPANCY_MAP).fillna("Other")

    if "lien_status" in df.columns:
        df["lien_status"] = df["lien_status"].map(LIEN_MAP).fillna("Other")

    if "conforming_loan_limit" in df.columns:
        df["conforming_loan_limit"] = (df["conforming_loan_limit"]
                                       .astype(str).str.strip()
                                       .map(CONFORMING_MAP))

    if "aus-1" in df.columns:
        df["aus-1"] = df["aus-1"].map(AUS_MAP).fillna("Other")

    if "submission_of_application" in df.columns:
        df["submission_of_application"] = (df["submission_of_application"]
                                           .map(SUBMISSION_MAP).fillna("Other"))

    if "applicant_credit_score_type" in df.columns:
        df["applicant_credit_score_type"] = (df["applicant_credit_score_type"]
                                             .map(CREDIT_SCORE_MAP).fillna("Other"))

    # derived_race / derived_sex / derived_ethnicity are already clean strings
    # state_code is already a string

    # Explicitly cast known numeric columns BEFORE detecting object columns.
    # These can arrive as object dtype due to NaN/mixed values during CSV load.
    FORCE_NUMERIC = [
            "loan_amount", "loan_term", "loan_to_value_ratio", "interest_rate",
            "property_value", "income", "debt_to_income_ratio", "applicant_age",
            "tract_minority_population_percent", "ffiec_msa_md_median_family_income",
            "tract_to_msa_income_percentage", "tract_owner_occupied_units",
            "tract_median_age_of_housing_units", "income_to_loan_ratio",
            "equity_ratio", "loan_amount_log", "income_log", "property_value_log",
            "ffiec_msa_md_median_family_income_log", "tract_owner_occupied_units_log",
            "conforming_loan_limit", "applicant_age_above_62",
            "co-applicant_age_above_62",
    ]
    for col in FORCE_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Safety check: skip any column with too many unique values (likely numeric sneaking through)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_cols = [c for c in cat_cols if c != "label"]
    safe_cat_cols = []
    for col in cat_cols:
        n_unique = df[col].nunique()
        if n_unique > 200:
            print(f"       [WARN] Skipping '{col}' ({n_unique} unique values) — force-casting to numeric.")
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            safe_cat_cols.append(col)

    print(f"       One-hot encoding {len(safe_cat_cols)} columns: {safe_cat_cols}")
    df = pd.get_dummies(df, columns=safe_cat_cols, drop_first=False, dtype=int)

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from raw columns.
    """

    # Income-to-loan ratio (core affordability signal)
    if "income" in df.columns and "loan_amount" in df.columns:
        income  = pd.to_numeric(df["income"],      errors="coerce")
        loanamt = pd.to_numeric(df["loan_amount"],  errors="coerce")
        df["income_to_loan_ratio"] = income / loanamt.replace(0, np.nan)

    # Property value to loan amount (equity signal)
    if "property_value" in df.columns and "loan_amount" in df.columns:
        propval = pd.to_numeric(df["property_value"], errors="coerce")
        loanamt = pd.to_numeric(df["loan_amount"],     errors="coerce")
        df["equity_ratio"] = (propval - loanamt) / propval.replace(0, np.nan)

    # Log transforms for skewed numeric columns
    for col in ["loan_amount", "income", "property_value",
                "ffiec_msa_md_median_family_income", "tract_owner_occupied_units"]:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").clip(lower=0)
            df[f"{col}_log"] = np.log1p(vals)

    return df


def coerce_all_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure every column except label is numeric.
    """
    for col in df.columns:
        if col != "label":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values:
        - Numeric columns: median
        - One - hot encoded columns are already 0 / 1 with no NaN after get_dummies
    """
    num_cols = [c for c in df.columns if c != "label"]
    medians  = df[num_cols].median()
    df[num_cols] = df[num_cols].fillna(medians)

    remaining = df.isnull().sum().sum()
    print(f"   Remaining NaN cells after fill: {remaining}")
    return df


def print_feature_summary(X: pd.DataFrame):
    print(f"\n  Final feature set: {X.shape[1]} features, {len(X):,} rows")
    print("  Sample features:")
    for col in list(X.columns[:8]) + ["..."] + list(X.columns[-3:]):
        if col != "...":
            print(f"    {col}")
        else:
            print(f"    ...")


def main():
    print("=" * 60)
    print("HMDA 2023 — Preprocessing Pipeline")
    print("=" * 60)

# ------------
# 1. Load
# -------------
print("\n[1/8] Loading data...")
df = load_data()

# ------------
# 2. Target
# -------------
print("\n[2/8] Mapping target variable...")
df = map_target(df)

# ----------------
# 3. DTI encoding
# ----------------
print("\n[3/8] Encoding debt_to_income_ratio (mixed types)...")
df = encode_dti(df)
print(f"       DTI sample: {df['debt_to_income_ratio'].dropna().head(5).tolist()}")

# ----------------
# 4. Age encoding
# ----------------
print("\n[4/8] Encoding applicant_age (string brackets → numeric)...")
df = encode_age(df)
df = encode_binary_flags(df)

# -----------------------
# 5. Feature engineering
# -----------------------

print("\n[5/8] Engineering features...")
df = feature_engineering(df)

# -----------------------
# 6. Categorical encoding
# -----------------------
print("\n[6/8] Encoding categorical variables...")
df = encode_categoricals(df)

# ------------------------
# 7. Coerce + fill missing
# ------------------------
print("\n[7/8] Coercing to numeric and filling missing values...")
df = coerce_all_numeric(df)
df = fill_missing(df)
print(f"       Shape after preprocessing: {df.shape}")

# ------------------
# 8. Split + scale
# -------------------
print("\n[8/8] Train/test split → scale (train only)...")
X = df.drop(columns=["label"])
y = df["label"]

print_feature_summary(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"\n       Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"       Train class dist:")
vc = y_train.value_counts().sort_index()
for k, v in vc.items():
    print(f"         {k} ({'Approved' if k==1 else 'Denied  '}): {v:,} ({v/len(y_train)*100:.1f}%)")

# Scale (fit on train only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# ------
# Save
# ------
print("\n[INFO] Saving processed data...")
feature_names = X.columns.tolist()

joblib.dump(X_train_scaled,  PROC_DIR / "X_train.pkl")
joblib.dump(y_train,  PROC_DIR / "y_train.pkl")
joblib.dump(X_test_scaled,  PROC_DIR / "X_test.pkl")
joblib.dump(y_test,         PROC_DIR / "y_test.pkl")
joblib.dump(scaler,         PROC_DIR / "scaler.pkl")
joblib.dump(feature_names,  PROC_DIR / "feature_names.pkl")

print(f"\n  Files saved to {PROC_DIR}/:")
for f in ["X_train.pkl","y_train.pkl","X_test.pkl","y_test.pkl",
          "scaler.pkl","feature_names.pkl"]:
    size = (PROC_DIR / f).stat().st_size / 1e6
    print(f"    {f:25s} {size:.1f} MB")

print(f"\n[DONE] Preprocessing complete.")
print("       Run train.py next.\n")


if __name__ == "__main__":
    main()