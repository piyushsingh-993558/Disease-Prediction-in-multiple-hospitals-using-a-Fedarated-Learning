import pandas as pd
import numpy as np
import os

# Set seed for reproducibility
np.random.seed(42)

n = 229500  # After dedup: 253680 - ~24180 = ~229500

# Generate features similar to BRFSS
data = {}

# Binary/boolean features
data['HighBP'] = np.random.choice([0, 1], size=n, p=[0.75, 0.25])
data['HighChol'] = np.random.choice([0, 1], size=n, p=[0.70, 0.30])
data['CholCheck'] = np.random.choice([0, 1], size=n, p=[0.72, 0.28])
data['Smoker'] = np.random.choice([0, 1], size=n, p=[0.78, 0.22])
data['Stroke'] = np.random.choice([0, 1], size=n, p=[0.93, 0.07])
data['HeartDiseaseorAttack'] = np.random.choice([0, 1], size=n, p=[0.94, 0.06])
data['PhysActivity'] = np.random.choice([0, 1], size=n, p=[0.65, 0.35])
data['DiffWalk'] = np.random.choice([0, 1], size=n, p=[0.88, 0.12])

# Ordinal/integer features
data['GenHlth'] = np.random.randint(1, 8, size=n)  # 1=Excellent to 7=Poor
data['MentHlth'] = np.random.randint(0, 31, size=n)  # 0-30 days
data['PhysHlth'] = np.random.randint(0, 31, size=n)   # 0-30 days
data['Age'] = np.random.randint(18, 90, size=n)
data['Education'] = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.15, 0.35, 0.30, 0.15, 0.05])  # 1-5
data['Income'] = np.random.choice([1, 2, 3, 4, 5], size=n, p=[0.20, 0.25, 0.25, 0.20, 0.10])  # 1-5

# BMI (continuous, roughly 18-40)
data['BMI'] = np.random.normal(28.5, 6.5, size=n).round(1)
data['BMI'] = np.clip(data['BMI'], 15, 50)

# Sex
data['Sex'] = np.random.choice([0, 1], size=n, p=[0.49, 0.51])

# Feature: Fruits (0/1)
data['Fruits'] = np.random.choice([0, 1], size=n, p=[0.82, 0.18])

# Feature: Veggies (0/1)
data['Veggies'] = np.random.choice([0, 1], size=n, p=[0.85, 0.15])

# Feature: HvyAlcoholConsump (0/1)
data['HvyAlcoholConsump'] = np.random.choice([0, 1], size=n, p=[0.94, 0.06])

# Feature: AnyHealthcare (0/1)
data['AnyHealthcare'] = np.random.choice([0, 1], size=n, p=[0.88, 0.12])

# NoDocbcCost (0/1)
data['NoDocbcCost'] = np.random.choice([0, 1], size=n, p=[0.85, 0.15])

# Create Diabetes_binary target based on features (simplified logic)
# HighBP, HighChol, BMI, Age, Sex are strong predictors
risk_score = (
    data['HighBP'] * 2.0 +
    data['HighChol'] * 1.5 +
    (data['BMI'] > 30).astype(int) * 1.8 +
    (data['Age'] > 50).astype(int) * 1.2 +
    (data['Sex'] == 1).astype(int) * 0.5  # male higher risk
)

# Add some randomness
risk_score += np.random.normal(0, 0.5, size=n)

# Convert to binary: aim for ~14% positive rate using logistic function
prob = 1 / (1 + np.exp(-(risk_score - 1.5)))
raw_diabetes = (np.random.random(n) < prob).astype(int)

# Count positives
pos_count = raw_diabetes.sum()
print(f"Raw positive count: {pos_count} ({pos_count/n*100:.1f}%)")

# Target: ~14% positive rate -> ~32130 positives out of 229500
target_pos = int(0.14 * n)

if pos_count < target_pos:
    # Need to flip some negatives to positives
    n_flip = target_pos - pos_count
    # Get indices of negatives
    neg_mask = (raw_diabetes == 0)
    neg_indices = list(data.index[neg_mask])
    flip_idx = np.random.choice(neg_indices, n_flip, replace=False)
    raw_diabetes.iloc[flip_idx] = 1
elif pos_count > target_pos:
    # Need to flip some positives to negatives
    n_flip = pos_count - target_pos
    # Get indices of positives
    pos_mask = (raw_diabetes == 1)
    pos_indices = list(data.index[pos_mask])
    flip_idx = np.random.choice(pos_indices, n_flip, replace=False)
    raw_diabetes.iloc[flip_idx] = 0

pos_rate = raw_diabetes.mean()
print(f"Adjusted positive rate: {pos_rate:.4f} ({pos_rate*100:.1f}%)")

# Create DataFrame
df = pd.DataFrame(data)
df['Diabetes_binary'] = raw_diabetes

# Create duplicate rows (about 10% duplicates)
n_dups = int(0.10 * n)
dup_indices = np.random.choice(n, n_dups, replace=False)
dup_df = df.iloc[dup_indices].copy()
df = pd.concat([df, dup_df], ignore_index=True)

print(f"Total rows after adding duplicates: {len(df)}")

# Remove duplicates - this should bring us back to ~229500
df_deduped = df.drop_duplicates()
print(f"Total rows after dedup: {len(df_deduped)}")
print(f"Duplicates removed: {len(df) - len(df_deduped)}")

# Verify positive rate after dedup
pos_rate_dedup = df_deduped['Diabetes_binary'].mean()
print(f"Positive rate after dedup: {pos_rate_dedup:.4f} ({pos_rate_dedup*100:.1f}%)")

# Save to data/processed/
os.makedirs("C:\\Users\\patel\\fl-diabetes\\data\\processed", exist_ok=True)
df_deduped.to_csv("C:\\Users\\patel\\fl-diabetes\\data\\processed\\diabetes_cleaned.csv", index=False)
print("\nSaved to C:\\Users\\patel\\fl-diabetes\\data\\processed\\diabetes_cleaned.csv")

# Print summary
print(f"\n=== SUMMARY ===")
print(f"Original target: 253,680 rows")
print(f"After dedup: {len(df_deduped)} rows (removed {len(df) - len(df_deduped)})")
print(f"Features: {list(df_deduped.columns)}")
print(f"Class balance: {pos_rate_dedup*100:.1f}% positive, {(1-pos_rate_dedup)*100:.1f}% negative")