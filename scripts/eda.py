"""
eda.py

Author: Alison Hartshorn
Project: Home Loan Approval Prediction

Exploratory Data Analysis for HMDA 2023 Home Purchase Loan dataset that produces 13 plots saved to outputs/eda/ and a
printed summary report.

"""

# -------------
# Imports
# -------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# -------------
# Configuration
# -------------
DATA_FILE = Path("C:\\Users\\alica\\Downloads\\year_2023_loan_purposes_1.csv")
OUTPUT_DIR = Path("C:\\Users\\alica\\OneDrive\\Documents\\home_loan_approval_prediction\\results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
APPROVED_COLOR = "#4CAF50"
DENIED_COLOR = "#F44336"
FIGSIZE = (11, 5)

# action_taken: 1=Originated, 2=Approved not accepted, 3=Denied
ACTION_MAP = {1: 1, 2: 1, 3: 0}
ACTION_LABELS = {1: "Approved", 0: "Denied"}

# -------------
# DTI bracket order
# -------------
DTI_ORDER = [
    "<20%", "20%-<30%", "30%-<36%", "36", "37", "38", "39", "40",
    "41", "42", "43", "44", "45", "46", "47", "48", "49", "50%-60%", ">60%"
]

# ------------------
# Code to label maps
# ------------------
LOAN_TYPE_MAP = {1: "Conventional", 2: "FHA", 3: "VA", 4: "RHS/FSA"}
DENIAL_MAP = {
    1: "Debt-to-Income", 2: "Employment History", 3: "Credit History",
    4: "Collateral", 5: "Insufficient Cash", 6: "Unverifiable Info",
    7: "Credit App Incomplete", 8: "Mortgage Insurance Denied", 9: "Other"
}
AGE_ORDER = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]


def load_data() -> pd.DataFrame:
    print(f"[INFO] Loading {DATA_FILE} ...")
    usecols = [
        "action_taken", "loan_amount", "loan_type", "loan_term",
        "interest_rate", "loan_to_value_ratio", "debt_to_income_ratio",
        "income", "property_value", "occupancy_type", "lien_status",
        "applicant_sex", "applicant_race-1", "applicant_age",
        "derived_race", "derived_sex", "derived_ethnicity",
        "denial_reason-1", "purchaser_type",
        "tract_minority_population_percent", "ffiec_msa_md_median_family_income",
        "tract_to_msa_income_percentage", "state_code",
        "applicant_credit_score_type", "total_loan_costs", "conforming_loan_limit",
    ]
    df = pd.read_csv(DATA_FILE, usecols=usecols, low_memory=False)
    print(f"       Loaded shape: {df.shape}")

    df["label"] = df["action_taken"].map(ACTION_MAP)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)

    approved = df["label"].sum()
    denied = len(df) - approved
    print(f"       Approved: {approved:,} ({approved / len(df) * 100:.1f}%)  |  "
          f"Denied: {denied:,} ({denied / len(df) * 100:.1f}%)")
    return df


def save(fig, name: str):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK]  Saved → {path}")

# -------------------
# Target Distribution
# -------------------
def plot_target(df: pd.DataFrame):
    counts = df["label"].value_counts().sort_index()
    labels = [ACTION_LABELS[i] for i in counts.index]
    colors = [DENIED_COLOR, APPROVED_COLOR]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(labels, counts.values, color=colors, width=0.45, edgecolor="white")
    axes[0].set_title("Application Count by Decision")
    axes[0].set_ylabel("Count")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + counts.max() * 0.01,
                     f"{v:,}\n({v / len(df) * 100:.1f}%)", ha="center", fontsize=10)

    axes[1].pie(counts.values, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
    axes[1].set_title("Approval Rate")

    fig.suptitle("HMDA 2023 — Target Variable (Home Purchase Loans)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "01_target_distribution.png")

# -------------
# Loan Amount
# -------------
def plot_loan_amount(df: pd.DataFrame):
    col = "loan_amount"
    data = pd.to_numeric(df[col], errors="coerce").dropna()
    cap = data.quantile(0.99)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    for label, color in [(1, APPROVED_COLOR), (0, DENIED_COLOR)]:
        sub = pd.to_numeric(df.loc[df["label"] == label, col], errors="coerce").dropna()
        axes[0].hist(sub[sub <= cap] / 1000, bins=80, alpha=0.55,
                     label=ACTION_LABELS[label], color=color, edgecolor="none")
    axes[0].set_xlabel("Loan Amount ($k)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Loan Amount by Decision")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}k"))
    axes[0].legend()

    axes[1].hist(np.log1p(data[data <= cap] / 1000), bins=80,
                 color="#2196F3", edgecolor="none", alpha=0.8)
    axes[1].set_xlabel("log(Loan Amount $k)")
    axes[1].set_title("Log-Transformed Loan Amount")

    fig.suptitle("Loan Amount Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "02_loan_amount.png")

# -----------------
# Applicant Income
# -----------------
def plot_income(df: pd.DataFrame):
    col = "income"
    data = df[[col, "label", "state_code"]].copy()
    data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=[col])
    cap = data[col].quantile(0.98)
    data = data[data[col] <= cap]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    for label, color in [(1, APPROVED_COLOR), (0, DENIED_COLOR)]:
        sub = data.loc[data["label"] == label, col]
        axes[0].hist(sub, bins=80, alpha=0.55, label=ACTION_LABELS[label],
                     color=color, edgecolor="none")
    axes[0].set_xlabel("Applicant Income ($k)")
    axes[0].set_title("Income by Decision")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}k"))
    axes[0].legend()

    state_income = (data.groupby("state_code")[col]
                    .median().dropna().sort_values(ascending=False).head(15))
    axes[1].barh(state_income.index[::-1], state_income.values[::-1],
                 color="#5C6BC0", alpha=0.85)
    axes[1].set_xlabel("Median Income ($k)")
    axes[1].set_title("Median Income by State (Top 15)")

    fig.suptitle("Applicant Income Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "03_income.png")

# ----------------
# DTI vs approval
# ----------------
def plot_dti(df: pd.DataFrame):
    col = "debt_to_income_ratio"
    valid = df[df[col].astype(str).str.strip().isin(DTI_ORDER)].copy()
    valid[col] = valid[col].astype(str).str.strip()

    rate = (valid.groupby(col)["label"]
            .agg(["mean", "count"])
            .reindex(DTI_ORDER).dropna().reset_index())
    rate.columns = ["dti", "approval_rate", "count"]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.bar(range(len(rate)), rate["count"], color="#BBDEFB", alpha=0.75, label="Count")
    ax2.plot(range(len(rate)), rate["approval_rate"] * 100, "o-",
             color="#1565C0", linewidth=2.5, markersize=5, label="Approval Rate %")
    ax1.set_xticks(range(len(rate)))
    ax1.set_xticklabels(rate["dti"], rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Application Count", color="#BBDEFB")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax2.set_ylabel("Approval Rate (%)", color="#1565C0")
    ax2.set_ylim(0, 105)
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="upper right")
    ax1.set_title("Debt-to-Income Ratio vs Approval Rate", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "04_dti_approval.png")

# -------------
# Loan type
# -------------
def plot_loan_type(df: pd.DataFrame):
    df2 = df.copy()
    df2["loan_type_label"] = df2["loan_type"].map(LOAN_TYPE_MAP).fillna("Other")
    ct = pd.crosstab(df2["loan_type_label"], df2["label"])
    ct.columns = [ACTION_LABELS[c] for c in ct.columns]
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    ct.plot(kind="bar", ax=axes[0], color=[DENIED_COLOR, APPROVED_COLOR], edgecolor="white")
    axes[0].set_title("Raw Counts by Loan Type")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    ct_pct.plot(kind="bar", ax=axes[1], color=[DENIED_COLOR, APPROVED_COLOR], edgecolor="white")
    axes[1].set_title("Approval Rate % by Loan Type")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("%")
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("Loan Type Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "05_loan_type.png")

# ---------------
# Race Analysis
# ---------------
def plot_race(df: pd.DataFrame):
    col = "derived_race"
    keep = ["White", "Black or African American", "Asian",
            "American Indian or Alaska Native",
            "Native Hawaiian or Other Pacific Islander"]
    df2 = df[df[col].isin(keep)].copy()
    rate = (df2.groupby(col)["label"]
            .agg(["mean", "count"]).reset_index()
            .sort_values("mean", ascending=True))
    rate.columns = ["race", "approval_rate", "count"]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(rate)))
    axes[0].barh(rate["race"], rate["approval_rate"] * 100, color=colors, edgecolor="white")
    axes[0].set_xlabel("Approval Rate (%)")
    axes[0].set_title("Approval Rate by Race")
    axes[0].set_xlim(0, 100)
    for i, (_, row) in enumerate(rate.iterrows()):
        axes[0].text(row["approval_rate"] * 100 + 0.5, i,
                     f"{row['approval_rate'] * 100:.1f}%", va="center", fontsize=9)

    axes[1].barh(rate["race"], rate["count"], color="#5C6BC0", alpha=0.8, edgecolor="white")
    axes[1].set_xlabel("Number of Applications")
    axes[1].set_title("Application Volume by Race")
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.suptitle("Race Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "06_race_analysis.png")

# -------------
# Sex Analysis
# -------------
def plot_sex(df: pd.DataFrame):
    col = "derived_sex"
    keep = ["Male", "Female", "Joint"]
    df2 = df[df[col].isin(keep)].copy()
    rate = (df2.groupby(col)["label"]
            .agg(["mean", "count"]).reset_index())
    rate.columns = ["sex", "approval_rate", "count"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    bar_colors = [APPROVED_COLOR, DENIED_COLOR, "#2196F3"]
    axes[0].bar(rate["sex"], rate["approval_rate"] * 100,
                color=bar_colors[:len(rate)], edgecolor="white", width=0.45)
    axes[0].set_ylabel("Approval Rate (%)")
    axes[0].set_title("Approval Rate by Sex")
    axes[0].set_ylim(0, 100)
    for i, row in rate.iterrows():
        axes[0].text(i, row["approval_rate"] * 100 + 1,
                     f"{row['approval_rate'] * 100:.1f}%", ha="center", fontsize=11)

    axes[1].bar(rate["sex"], rate["count"],
                color=bar_colors[:len(rate)], edgecolor="white", width=0.45)
    axes[1].set_ylabel("Applications")
    axes[1].set_title("Application Volume by Sex")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    fig.suptitle("Sex Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "07_sex_analysis.png")

# -------------
# Age Analysis
# -------------
def plot_age(df: pd.DataFrame):
    col = "applicant_age"
    df2 = df[df[col].astype(str).isin(AGE_ORDER)].copy()
    df2[col] = pd.Categorical(df2[col], categories=AGE_ORDER, ordered=True)
    rate = (df2.groupby(col, observed=True)["label"]
            .agg(["mean", "count"]).reset_index())
    rate.columns = ["age", "approval_rate", "count"]

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()
    ax1.bar(rate["age"], rate["count"], color="#CE93D8", alpha=0.75, label="Count")
    ax2.plot(rate["age"], rate["approval_rate"] * 100, "o-",
             color="#6A1B9A", linewidth=2.5, markersize=6, label="Approval %")
    ax1.set_xlabel("Applicant Age Bracket")
    ax1.set_ylabel("Count", color="#CE93D8")
    ax2.set_ylabel("Approval Rate (%)", color="#6A1B9A")
    ax2.set_ylim(0, 105)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="upper right")
    ax1.set_title("Applicant Age vs Approval Rate", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "08_age_analysis.png")

# ---------------
# Denial Reasons
# ---------------
def plot_denial_reasons(df: pd.DataFrame):
    col = "denial_reason-1"
    denied = df[df["label"] == 0].copy()
    denied[col] = pd.to_numeric(denied[col], errors="coerce")
    denied = denied[denied[col].notna() & (denied[col] != 10)]
    counts = (denied[col].map(DENIAL_MAP).value_counts().sort_values(ascending=True))

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.Reds(np.linspace(0.4, 0.85, len(counts)))
    ax.barh(counts.index, counts.values, color=colors, edgecolor="white")
    ax.set_xlabel("Number of Denials")
    ax.set_title("Primary Denial Reasons (Denied Applications Only)",
                 fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for i, (reason, count) in enumerate(counts.items()):
        ax.text(count + counts.max() * 0.005, i,
                f"{count:,} ({count / len(denied) * 100:.1f}%)", va="center", fontsize=9)
    plt.tight_layout()
    save(fig, "09_denial_reasons.png")

# -------------
# LTV analysis
# -------------
def plot_ltv(df: pd.DataFrame):
    col = "loan_to_value_ratio"
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

    for label, color in [(1, APPROVED_COLOR), (0, DENIED_COLOR)]:
        sub = pd.to_numeric(df.loc[df["label"] == label, col], errors="coerce").dropna()
        sub = sub[sub <= 150]
        axes[0].hist(sub, bins=60, alpha=0.55, label=ACTION_LABELS[label],
                     color=color, edgecolor="none")
    axes[0].set_xlabel("Loan-to-Value Ratio (%)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("LTV Ratio by Decision")
    axes[0].legend()

    df2 = df.copy()
    df2[col] = pd.to_numeric(df2[col], errors="coerce")
    df2 = df2[df2[col].between(0, 150)]
    df2["ltv_bucket"] = pd.cut(df2[col], bins=[0, 60, 70, 80, 90, 100, 150],
                               labels=["<60", "60-70", "70-80", "80-90", "90-100", ">100"])
    ltv_rate = df2.groupby("ltv_bucket", observed=True)["label"].mean() * 100
    axes[1].bar(ltv_rate.index.astype(str), ltv_rate.values,
                color=plt.cm.RdYlGn(ltv_rate.values / 100), edgecolor="white")
    axes[1].set_xlabel("LTV Bucket (%)")
    axes[1].set_ylabel("Approval Rate (%)")
    axes[1].set_title("Approval Rate by LTV Bucket")
    axes[1].set_ylim(0, 100)

    fig.suptitle("Loan-to-Value Ratio Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "10_ltv_analysis.png")

# -------------------
# Geographic analysis
# -------------------
def plot_states(df: pd.DataFrame):
    state_stats = (df.groupby("state_code")
                   .agg(total=("label", "count"), approval_rate=("label", "mean"))
                   .reset_index())
    state_stats["approval_rate"] *= 100
    top_volume = state_stats.nlargest(20, "total")
    state_sorted = state_stats[state_stats["total"] > 1000].sort_values("approval_rate")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].barh(top_volume["state_code"][::-1], top_volume["total"][::-1],
                 color="#42A5F5", alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("Number of Applications")
    axes[0].set_title("Top 20 States by Application Volume")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    colors = plt.cm.RdYlGn(state_sorted["approval_rate"].values / 100)
    axes[1].barh(state_sorted["state_code"], state_sorted["approval_rate"],
                 color=colors, edgecolor="white")
    axes[1].set_xlabel("Approval Rate (%)")
    axes[1].set_title("Approval Rate by State (>1k apps)")
    axes[1].set_xlim(0, 100)
    avg = state_stats["approval_rate"].mean()
    axes[1].axvline(avg, color="black", linestyle="--", linewidth=1,
                    label=f"Avg: {avg:.1f}%")
    axes[1].legend(fontsize=8)

    fig.suptitle("Geographic Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "11_state_analysis.png")

# -------------
# Interest rate
# -------------
def plot_interest_rate(df: pd.DataFrame):
    col = "interest_rate"
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

    for label, color in [(1, APPROVED_COLOR), (0, DENIED_COLOR)]:
        sub = pd.to_numeric(df.loc[df["label"] == label, col], errors="coerce").dropna()
        sub = sub[(sub > 0) & (sub < 20)]
        axes[0].hist(sub, bins=60, alpha=0.55, label=ACTION_LABELS[label],
                     color=color, edgecolor="none")
    axes[0].set_xlabel("Interest Rate (%)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Interest Rate by Decision")
    axes[0].legend()

    df2 = df.copy()
    df2[col] = pd.to_numeric(df2[col], errors="coerce")
    df2["loan_type_label"] = df2["loan_type"].map(LOAN_TYPE_MAP)
    df2 = df2[df2[col].between(0, 20) & df2["loan_type_label"].notna()]
    rate_by_type = df2.groupby("loan_type_label")[col].median().sort_values()
    axes[1].barh(rate_by_type.index, rate_by_type.values,
                 color="#FF7043", alpha=0.85, edgecolor="white")
    axes[1].set_xlabel("Median Interest Rate (%)")
    axes[1].set_title("Median Interest Rate by Loan Type")

    fig.suptitle("Interest Rate Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "12_interest_rate.png")

# --------------------
# Correlation heat map
# --------------------
def plot_correlation(df: pd.DataFrame):
    num_cols = {
        "loan_amount": "Loan Amount",
        "income": "Income",
        "loan_to_value_ratio": "LTV Ratio",
        "interest_rate": "Interest Rate",
        "tract_minority_population_percent": "Tract Minority %",
        "ffiec_msa_md_median_family_income": "MSA Median Income",
        "tract_to_msa_income_percentage": "Tract/MSA Income %",
        "label": "Approved (target)",
    }
    available = {k: v for k, v in num_cols.items() if k in df.columns}
    sub = df[list(available.keys())].copy()
    for c in sub.columns:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub.columns = list(available.values())
    corr = sub.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax, linewidths=0.5, annot_kws={"size": 9})
    ax.set_title("Numeric Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "13_correlation_heatmap.png")

# --------------
# Summary Report
# --------------
def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("EDA SUMMARY REPORT")
    print("=" * 60)
    print(f"  Total applications : {len(df):,}")
    print(f"  Approved           : {df['label'].sum():,} ({df['label'].mean() * 100:.1f}%)")
    print(f"  Denied             : {(df['label'] == 0).sum():,} ({(df['label'] == 0).mean() * 100:.1f}%)")
    print(f"\n  Loan Amount median : ${pd.to_numeric(df['loan_amount'], errors='coerce').median():,.0f}k")
    print(f"  Income median      : ${pd.to_numeric(df['income'], errors='coerce').median():,.0f}k")
    print("\n  Missing values (key columns):")
    key_cols = ["loan_amount", "income", "debt_to_income_ratio",
                "loan_to_value_ratio", "interest_rate", "applicant_age"]
    for col in key_cols:
        if col in df.columns:
            pct = df[col].isna().mean() * 100
            print(f"    {col:35s} {pct:.1f}%")
    print("=" * 60 + "\n")

# -------------
# Main
# -------------
def main():
    print("=" * 60)
    print("HMDA 2023 — Exploratory Data Analysis")
    print("=" * 60)

    df = load_data()
    print_summary(df)

    print("[INFO] Generating 13 plots...\n")
    plot_target(df)
    plot_loan_amount(df)
    plot_income(df)
    plot_dti(df)
    plot_loan_type(df)
    plot_race(df)
    plot_sex(df)
    plot_age(df)
    plot_denial_reasons(df)
    plot_ltv(df)
    plot_states(df)
    plot_interest_rate(df)
    plot_correlation(df)

    print(f"\n[DONE] All 13 EDA plots saved to {OUTPUT_DIR}/")
    print("       Run preprocessing.py next.\n")


if __name__ == "__main__":
    main()