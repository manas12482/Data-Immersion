"""
=============================================================
  DATA IMMERSION & WRANGLING — TASK 1
  Data Cleaning & Transformation Script
  Author  : Data Analytics Intern
  Dataset : raw_customer_data.csv
  Date    : 2026-03-15
=============================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 1 — LOAD & INITIAL PROFILING
# ─────────────────────────────────────────────

print("=" * 60)
print("STEP 1 — LOADING DATA & INITIAL PROFILING")
print("=" * 60)

df = pd.read_csv("raw_customer_data.csv")

print(f"\n[INFO] Shape            : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"[INFO] Duplicate rows   : {df.duplicated().sum()}")
print("\n[INFO] Missing values per column:")
missing = df.isnull().sum()
missing_pct = (df.isnull().mean() * 100).round(2)
print(pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
      [missing > 0].to_string())

print("\n[INFO] Data types:")
print(df.dtypes.to_string())

print("\n[INFO] Sample problematic values:")
print("  gender unique  :", df['gender'].dropna().unique())
print("  membership_tier:", df['membership_tier'].dropna().unique())
print("  is_active      :", df['is_active'].dropna().unique())


# ─────────────────────────────────────────────
# STEP 2 — REMOVE DUPLICATES
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2 — REMOVING DUPLICATES")
print("=" * 60)

before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"[INFO] Removed {before - after} duplicate rows. Rows remaining: {after}")


# ─────────────────────────────────────────────
# STEP 3 — STANDARDISE DATE FORMATS
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3 — STANDARDISING DATE FORMATS")
print("=" * 60)

def parse_flexible_date(val):
    """Try multiple date formats and return a uniform datetime."""
    if pd.isna(val):
        return pd.NaT
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(str(val), fmt)
        except ValueError:
            continue
    return pd.NaT

for col in ['date_of_birth', 'registration_date', 'last_purchase_date']:
    original_nulls = df[col].isna().sum()
    df[col] = df[col].apply(parse_flexible_date)
    new_nulls = df[col].isna().sum()
    failed = new_nulls - original_nulls
    print(f"[INFO] '{col}' — parsed successfully. Unparseable: {failed}")

print("[INFO] All date columns now in datetime format.")


# ─────────────────────────────────────────────
# STEP 4 — FEATURE ENGINEERING
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4 — FEATURE ENGINEERING")
print("=" * 60)

reference_date = datetime(2026, 3, 15)

# Customer Age
df['customer_age'] = df['date_of_birth'].apply(
    lambda dob: int((reference_date - dob).days / 365.25) if pd.notna(dob) else np.nan
)
print(f"[INFO] 'customer_age' created. Range: {df['customer_age'].min():.0f}–{df['customer_age'].max():.0f} yrs")

# Tenure in days
df['tenure_days'] = df['registration_date'].apply(
    lambda reg: (reference_date - reg).days if pd.notna(reg) else np.nan
)
print(f"[INFO] 'tenure_days' created. Range: {df['tenure_days'].min():.0f}–{df['tenure_days'].max():.0f} days")

# Days since last purchase
df['days_since_last_purchase'] = df['last_purchase_date'].apply(
    lambda lp: (reference_date - lp).days if pd.notna(lp) else np.nan
)
print(f"[INFO] 'days_since_last_purchase' created.")

# Average purchase value (placeholder — would use transaction table in production)
df['avg_purchase_value'] = (df['total_purchases'] / (df['tenure_days'] / 30)).round(2)
print(f"[INFO] 'avg_purchase_value' (monthly avg) created.")


# ─────────────────────────────────────────────
# STEP 5 — CATEGORICAL STANDARDISATION
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5 — STANDARDISING CATEGORICAL FIELDS")
print("=" * 60)

# Gender
gender_map = {
    'male': 'Male', 'm': 'Male', 'M': 'Male',
    'female': 'Female', 'f': 'Female', 'F': 'Female'
}
df['gender'] = df['gender'].map(lambda x: gender_map.get(str(x).strip(), x) if pd.notna(x) else np.nan)
df['gender'] = df['gender'].apply(lambda x: x if x in ['Male', 'Female'] else np.nan)
print(f"[INFO] 'gender' standardised: {df['gender'].value_counts().to_dict()}")

# Membership tier
df['membership_tier'] = df['membership_tier'].apply(
    lambda x: str(x).strip().capitalize() if pd.notna(x) else np.nan
)
valid_tiers = ['Silver', 'Gold', 'Platinum']
df['membership_tier'] = df['membership_tier'].apply(
    lambda x: x if x in valid_tiers else np.nan
)
print(f"[INFO] 'membership_tier' standardised: {df['membership_tier'].value_counts().to_dict()}")

# is_active → boolean
def to_bool(val):
    if pd.isna(val):
        return np.nan
    v = str(val).strip().lower()
    if v in ('true', '1', 'yes'):
        return True
    if v in ('false', '0', 'no'):
        return False
    return np.nan

df['is_active'] = df['is_active'].apply(to_bool)
print(f"[INFO] 'is_active' standardised: {df['is_active'].value_counts().to_dict()}")


# ─────────────────────────────────────────────
# STEP 6 — HANDLE OUTLIERS & INVALID VALUES
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6 — HANDLING OUTLIERS & INVALID VALUES")
print("=" * 60)

# Negative total_purchases → set to NaN (data entry error)
neg_mask = df['total_purchases'] < 0
print(f"[INFO] Negative 'total_purchases' found: {neg_mask.sum()} rows → set to NaN")
df.loc[neg_mask, 'total_purchases'] = np.nan

# Customer age sanity check
invalid_age = (df['customer_age'] < 18) | (df['customer_age'] > 100)
print(f"[INFO] Invalid 'customer_age' (< 18 or > 100): {invalid_age.sum()} rows → set to NaN")
df.loc[invalid_age, 'customer_age'] = np.nan


# ─────────────────────────────────────────────
# STEP 7 — IMPUTE MISSING VALUES
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 7 — IMPUTING MISSING VALUES")
print("=" * 60)

# Numeric: median imputation
for col in ['total_purchases', 'customer_age']:
    n_missing = df[col].isna().sum()
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f"[INFO] '{col}' — {n_missing} missing values filled with median ({median_val:.2f})")

# Categorical: mode imputation
for col in ['gender', 'membership_tier', 'city']:
    n_missing = df[col].isna().sum()
    mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
    df[col] = df[col].fillna(mode_val)
    print(f"[INFO] '{col}' — {n_missing} missing values filled with mode ('{mode_val}')")

# Boolean is_active: default False
n_missing = df['is_active'].isna().sum()
df['is_active'] = df['is_active'].fillna(False)
print(f"[INFO] 'is_active' — {n_missing} missing values filled with False")


# ─────────────────────────────────────────────
# STEP 8 — FINAL VALIDATION & SAVE
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 8 — FINAL VALIDATION")
print("=" * 60)

print(f"[INFO] Final shape     : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"[INFO] Remaining nulls : {df.isnull().sum().sum()}")
print(f"[INFO] Duplicates      : {df.duplicated().sum()}")
print("\n[INFO] Final columns   :", list(df.columns))
print("\n[INFO] Sample (first 3 rows):")
print(df.head(3).to_string())

output_path = "cleaned_customer_data.csv"
df.to_csv(output_path, index=False)
print(f"\n[SUCCESS] Cleaned dataset saved → {output_path}")
print("=" * 60)
