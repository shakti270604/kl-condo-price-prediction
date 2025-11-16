import pandas as pd

# Load your cleaned dataset
df = pd.read_csv("kl_condominium_final_cleaned.csv")

# Check unique values under location_clean
print("Unique values in location_clean column:")
print(df["location_clean"].unique())

# Count occurrences of each location
print("\nLocation counts:")
print(df["location_clean"].value_counts())

# ============================================================
# STEP 4: STANDARDIZE TEXT VALUES (e.g., merge 'klcc' and 'KLCC')
# ============================================================

# Identify all object (categorical) columns
categorical_cols = df.select_dtypes(include=['object']).columns
print("🧾 Categorical columns found:", list(categorical_cols))

# 1️⃣ Convert all text values to uppercase and strip whitespace
for col in categorical_cols:
    df[col] = df[col].astype(str).str.upper().str.strip()

print("\n✅ All categorical text columns standardized to uppercase.")

# 2️⃣ Verify the fix worked for 'location_clean'
if "location_clean" in df.columns:
    print("\n📍 Unique locations after standardization:")
    print(sorted(df["location_clean"].unique()))

# ============================================================
# STEP 1: DATA CLEANING — INSPECTION
# ============================================================
from IPython.display import display
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("kl_condominium_final_cleaned.csv")

# Display basic info
print("✅ Dataset loaded successfully!\n")
print("Shape:", df.shape)
print("\n--- COLUMNS ---")
print(df.columns.tolist())

# Check data types and non-null counts
print("\n--- INFO ---")
df.info()

# Quick look at first few rows
print("\n--- SAMPLE DATA ---")
display(df.head())

# Check for missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# Summary statistics for numeric columns
print("\n--- SUMMARY STATISTICS ---")
display(df.describe())

# Unique values for object columns
print("\n--- UNIQUE VALUES (categorical) ---")
for col in df.select_dtypes(include="object").columns:
    print(f"{col}: {df[col].nunique()} unique values")

# ============================================================
# STEP 2: HANDLE MISSING VALUES
# ============================================================

# 1️⃣ Show total missing values per column
print("🔍 Missing Values Summary:")
print(df.isnull().sum())

# 2️⃣ View percentage of missing values
missing_percent = (df.isnull().sum() / len(df)) * 100
print("\n📊 Missing Values (%):")
print(missing_percent)

# 3️⃣ Example handling strategies:

## Numeric columns — fill with median (robust to outliers)
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        median_value = df[col].median()
        df[col].fillna(median_value, inplace=True)
        print(f"✅ Filled missing numeric values in '{col}' with median: {median_value}")

## Categorical columns — fill with mode (most frequent value)
for col in df.select_dtypes(include=['object']).columns:
    if df[col].isnull().sum() > 0:
        mode_value = df[col].mode()[0]
        df[col].fillna(mode_value, inplace=True)
        print(f"✅ Filled missing categorical values in '{col}' with mode: {mode_value}")

# 4️⃣ Confirm no missing values remain
print("\n✅ Missing values after cleaning:")
print(df.isnull().sum().sum(), "missing cells remain.")

# ============================================================
# STEP 3: REMOVE DUPLICATES
# ============================================================

# 1️⃣ Count duplicates before removal
duplicate_count = df.duplicated().sum()
print(f"🔍 Number of duplicate rows before removal: {duplicate_count}")

# 2️⃣ Display duplicate examples (if any)
if duplicate_count > 0:
    print("\n--- SAMPLE DUPLICATES ---")
    display(df[df.duplicated()].head())

# 3️⃣ Remove duplicates
df.drop_duplicates(inplace=True)

# 4️⃣ Count after removal
print(f"\n✅ Number of duplicate rows after removal: {df.duplicated().sum()}")
print(f"📉 Total rows remaining: {len(df)}")

# ============================================================
# STEP 4: STANDARDIZE TEXT VALUES (e.g., merge 'klcc' and 'KLCC')
# ============================================================

# Identify all object (categorical) columns
categorical_cols = df.select_dtypes(include=['object']).columns
print("🧾 Categorical columns found:", list(categorical_cols))

# 1️⃣ Convert all text values to uppercase and strip whitespace
for col in categorical_cols:
    df[col] = df[col].astype(str).str.upper().str.strip()

print("\n✅ All categorical text columns standardized to uppercase.")

# 2️⃣ Verify the fix worked for 'location_clean'
if "location_clean" in df.columns:
    print("\n📍 Unique locations after standardization:")
    print(sorted(df["location_clean"].unique()))

# ============================================================
# STEP 5: OUTLIER DETECTION & TREATMENT
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns

# Select numeric columns for outlier analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns
print("📊 Numeric columns:", list(numeric_cols))

# 1️⃣ Detect outliers using the IQR method
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    before = data.shape[0]
    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    after = data.shape[0]
    removed = before - after
    print(f"🧹 {column}: Removed {removed} outliers (bounds: {lower_bound:.2f} – {upper_bound:.2f})")
    return data

# 2️⃣ Apply to important numeric columns (adjust as needed)
for col in ["size_sqft", "price_myr"]:
    if col in df.columns:
        df = remove_outliers_iqr(df, col)

# 3️⃣ Visualize before/after distribution for verification
plt.figure(figsize=(8, 5))
sns.boxplot(df["size_sqft"])
plt.title("📦 Boxplot After Outlier Removal — size_sqft")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(df["price_myr"])
plt.title("📦 Boxplot After Outlier Removal — price_myr")
plt.show()

print("\n✅ Outlier cleaning complete.")

# ============================================================
# STEP 7: ENCODING CATEGORICAL VARIABLES — ONE-HOT ENCODING
# ============================================================

# 1️⃣ Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
print("📋 Categorical columns to encode:", list(categorical_cols))

# 2️⃣ Perform One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

print("\n✅ One-Hot Encoding complete!")
print("New shape after encoding:", df_encoded.shape)

# 3️⃣ Display encoded feature sample
print("\n--- SAMPLE ENCODED COLUMNS ---")
display(df_encoded.head(3))

# 4️⃣ Check the number of new features created
added_features = df_encoded.shape[1] - df.shape[1]
print(f"\n🧮 Added {added_features} new columns after encoding.")

# ============================================================
# STEP 8: FEATURE SCALING — STANDARDIZATION
# ============================================================

from sklearn.preprocessing import StandardScaler

# We now have df_encoded from previous step (Step 7)
# We'll prepare TWO versions of the dataset:
#  - df_unscaled → for tree-based models (Random Forest, XGBoost)
#  - df_scaled   → for regression and ANN

# 1️⃣ Create unscaled copy
df_unscaled = df_encoded.copy()

# 2️⃣ Identify features to scale (exclude target column)
features_to_scale = [col for col in df_encoded.columns if col != "price_myr"]

# 3️⃣ Initialize scaler
scaler = StandardScaler()

# 4️⃣ Create scaled version
df_scaled = df_encoded.copy()
df_scaled[features_to_scale] = scaler.fit_transform(df_encoded[features_to_scale])

# 5️⃣ Display scaling results
print("\n✅ Feature scaling complete!")
print("Mean of scaled features (≈0):")
print(df_scaled[features_to_scale].mean().round(2).head())

print("\nStandard deviation (≈1):")
print(df_scaled[features_to_scale].std().round(2).head())

# ============================================================
# STEP 9: TRAIN-TEST SPLIT + K-FOLD SETUP
# ============================================================

from sklearn.model_selection import train_test_split, KFold

# --- Unscaled version ---
X_unscaled = df_unscaled.drop(columns=["price_myr"], errors="ignore")
y_unscaled = df_unscaled["price_myr"]

X_train_unscaled, X_test_unscaled, y_train_unscaled, y_test_unscaled = train_test_split(
    X_unscaled, y_unscaled, test_size=0.2, random_state=42
)

# --- Scaled version ---
X_scaled = df_scaled.drop(columns=["price_myr"], errors="ignore")
y_scaled = df_scaled["price_myr"]

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

# --- K-Fold setup ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n✅ Data split complete!")
print(f"Unscaled: {X_train_unscaled.shape} train, {X_test_unscaled.shape} test")
print(f"Scaled:   {X_train_scaled.shape} train, {X_test_scaled.shape} test")

# ============================================================
# STEP 10: SAVE PREPROCESSED DATASETS
# ============================================================

# --- Unscaled ---
train_unscaled = pd.concat([X_train_unscaled, y_train_unscaled], axis=1)
test_unscaled  = pd.concat([X_test_unscaled, y_test_unscaled], axis=1)
train_unscaled.rename(columns={y_train_unscaled.name: "Target"}, inplace=True)
test_unscaled.rename(columns={y_test_unscaled.name: "Target"}, inplace=True)
train_unscaled.to_csv("train_preprocessed_unscaled.csv", index=False)
test_unscaled.to_csv("test_preprocessed_unscaled.csv", index=False)

# --- Scaled ---
train_scaled = pd.concat([X_train_scaled, y_train_scaled], axis=1)
test_scaled  = pd.concat([X_test_scaled, y_test_scaled], axis=1)
train_scaled.rename(columns={y_train_scaled.name: "Target"}, inplace=True)
test_scaled.rename(columns={y_test_scaled.name: "Target"}, inplace=True)
train_scaled.to_csv("train_preprocessed_scaled.csv", index=False)
test_scaled.to_csv("test_preprocessed_scaled.csv", index=False)

print("\n✅ Preprocessed data saved successfully!")
print("train_preprocessed_unscaled.csv →", train_unscaled.shape)
print("test_preprocessed_unscaled.csv →", test_unscaled.shape)
print("train_preprocessed_scaled.csv →", train_scaled.shape)
print("test_preprocessed_scaled.csv →", test_scaled.shape)