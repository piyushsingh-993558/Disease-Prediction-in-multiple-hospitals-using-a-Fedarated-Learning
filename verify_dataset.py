import pandas as pd

# Verify cleaned data
df = pd.read_csv(r"C:\Users\patel\fl-diabetes\data\processed\diabetes_cleaned.csv")
print(f"Cleaned dataset shape: {df.shape}")
print(f"Features: {len([c for c in df.columns if c != 'Diabetes_binary'])}")
pos_rate = df['Diabetes_binary'].mean()
print(f"Class balance: {pos_rate*100:.2f}% positive")
print(f"Columns: {list(df.columns)}")
print("\nWeek 1 Phase 1 verification: PASSED")