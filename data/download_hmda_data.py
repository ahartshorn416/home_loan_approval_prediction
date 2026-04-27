"""
download_hmda_data.py

Author: Alison Hartshorn
Project Name: Home loan prediction

Downloads the HMDA 2023 nationwide loan-level dataset from the CFPB API.
Saves a filtered subset (home purchase loans only) to data/raw/.

HMDA Data Source: https://ffiec.cfpb.gov/data-browser/
API Docs:        https://ffiec.cfpb.gov/documentation/api/data-browser/
"""
# -----------------------------
# Imports
# -----------------------------
import os
import requests
import pandas as pd
from pathlib import Path

# -----------------------------
# Configs
# -----------------------------
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RAW_DIR / "hmda_2023_home_purchase.csv"

# HMDA Data Browser API — 2023 nationwide, home purchase loans (loan_purpose=1)
# action_taken: 1=originated, 2=approved not accepted, 3=denied, 4=withdrawn,
#               5=incomplete, 6=purchased, 7=preapproval denied, 8=preapproval approved
API_URL = (
    "https://ffiec.cfpb.gov/data-browser-api/view/csv"
    "?years=2023"
    "&loan_purposes=1"          # Home purchase only
    "&actions_taken=1,2,3"      # Originated, approved not accepted, denied
)

# Columns to keep
KEEP_COLS = [
    "action_taken",
    "loan_amount",
    "loan_term",
    "loan_type",
    "property_type",
    "occupancy_type",
    "applicant_income",
    "applicant_race-1",
    "applicant_sex",
    "applicant_age",
    "co-applicant_present",
    "debt_to_income_ratio",
    "combined_loan_to_value_ratio",
    "interest_rate",
    "total_loan_costs",
    "lien_status",
    "hoepa_status",
    "manufactured_home_land_property_interest",
    "census_tract",
    "county_code",
    "tract_population",
    "tract_minority_population_percent",
    "ffiec_msa_md_median_family_income",
    "tract_to_msa_income_percentage",
    "tract_owner_occupied_units",
    "applicant_credit_score_type",
    "denial_reason-1",
]


def download_hmda():
    print("=" * 60)
    print("HMDA 2023 Home Purchase Loan Data Downloader")
    print("=" * 60)

    if OUTPUT_FILE.exists():
        print(f"[SKIP] File already exists: {OUTPUT_FILE}")
        print("       Delete it to re-download.")
        return

    print(f"\n[INFO] Fetching data from CFPB API...")
    print(f"       URL: {API_URL}\n")
    print("       This may take several minutes (dataset is large)...\n")

    try:
        response = requests.get(API_URL, stream=True, timeout=300)
        response.raise_for_status()

        # Stream to disk first, then filter
        raw_path = RAW_DIR / "hmda_2023_raw_download.csv"
        total = 0
        with open(raw_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                total += len(chunk)
                print(f"\r       Downloaded: {total / 1e6:.1f} MB", end="", flush=True)

        print(f"\n\n[OK]  Download complete: {total / 1e6:.1f} MB")

    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\nManual download instructions:")
        print("  1. Go to: https://ffiec.cfpb.gov/data-browser/data/2023?category=nationwide")
        print("  2. Filter: Loan Purpose = Home Purchase, Action Taken = 1,2,3")
        print("  3. Click 'Download CSV'")
        print(f"  4. Save to: {RAW_DIR}/hmda_2023_raw_download.csv")
        return

    # Filter to relevant columns
    print("[INFO] Filtering columns...")
    try:
        df = pd.read_csv(raw_path, low_memory=False)
        print(f"       Raw shape: {df.shape}")

        # Keep only columns that exist in the file
        available = [c for c in KEEP_COLS if c in df.columns]
        missing = [c for c in KEEP_COLS if c not in df.columns]

        if missing:
            print(f"[WARN] These columns were not found and will be skipped:")
            for c in missing:
                print(f"       - {c}")

        df = df[available]
        print(f"       Filtered shape: {df.shape}")

        df.to_csv(OUTPUT_FILE, index=False)
        print(f"[OK]  Saved filtered data → {OUTPUT_FILE}")

        # Clean up raw download
        raw_path.unlink()
        print(f"[OK]  Removed raw download file.")

    except Exception as e:
        print(f"[ERROR] Failed to process downloaded file: {e}")
        return

    # Summary
    print("\n── Dataset Summary ──────────────────────────────────────────")
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {len(df.columns)}")
    print(f"\n  action_taken distribution:")
    dist = df["action_taken"].value_counts().sort_index()
    labels = {1: "Originated (approved)", 2: "Approved/not accepted", 3: "Denied"}
    for code, count in dist.items():
        label = labels.get(code, str(code))
        print(f"    {code} - {label}: {count:,} ({count/len(df)*100:.1f}%)")
    print("─" * 60)
    print("\n[DONE] Run 02_eda.py next.\n")


if __name__ == "__main__":
    download_hmda()