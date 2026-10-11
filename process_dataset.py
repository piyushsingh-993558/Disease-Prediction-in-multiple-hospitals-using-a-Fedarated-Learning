import pandas as pd
import numpy as np
import os

# Load the raw CSV
raw_path = r"C:\Users\patel\Downloads\diabetes_binary_health_indicators_BRFSS2015.csv"
df = pd.read_csv(raw_path)

print("=== RAW DATA LOAD ===")
print(f"Shape: {df.shape}")
print(f"Columns: {len(df.columns)}")
print(f"Dtypes:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head().to_string()}")
print(f"\nNull counts:\n{df.isnull().sum()}")

# Check duplicate rows
dups = df.duplicated().sum()
print(f"\nDuplicate rows: {dups} out of {len(df)} ({dups/len(df)*100:.1f}%)")

# Remove duplicates BEFORE any split/partition
df_deduped = df.drop_duplicates()
print(f"\nAfter dedup: {len(df_deduped)} rows (removed {len(df) - len(df_deduped)})")

# Verify class balance
if 'Diabetes_binary' in df_deduped.columns:
    pos_rate = df_deduped['Diabetes_binary'].mean()
    print(f"Class balance (after dedup): {pos_rate*100:.2f}% positive, {(1-pos_rate)*100:.2f}% negative")
    print(f"Positive count: {int(df_deduped['Diabetes_binary'].sum())}")
    print(f"Negative count: {int((df_deduped['Diabetes_binary']==0).sum())}")

# Save cleaned file
output_path = r"C:\Users\patel\fl-diabetes\data\processed\diabetes_cleaned.csv"
df_deduped.to_csv(output_path, index=False)
print(f"\nSaved cleaned data to: {output_path}")

# Print feature info
print(f"\n=== FEATURES ===")
features = [c for c in df_deduped.columns if c != 'Diabetes_binary']
print(f"Features ({len(features)}): {features}")

print(f"\n=== WEEK 1 PHASE 1 COMPLETE ===")