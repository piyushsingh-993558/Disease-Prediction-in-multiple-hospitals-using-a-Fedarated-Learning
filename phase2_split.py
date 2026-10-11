import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load cleaned data
df = pd.read_csv(r"C:\Users\patel\fl-diabetes\data\processed\diabetes_cleaned.csv")

print("=== PHASE 2: Train / Global Test Split ===")

# Step 1: Hold out global test set (~20%), stratified on label
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']

X_train_global, X_test, y_train_global, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_df = pd.concat([X_train_global, y_train_global], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

print(f"Global train pool: {len(train_df)} rows (80%)")
print(f"Global test holdout: {len(test_df)} rows (20%)")
print(f"Test set positive rate: {test_df['Diabetes_binary'].mean()*100:.2f}%")

# Step 2: Partition the 80% train pool across 5 hospitals
n_clients = 5
train_indices = np.arange(len(train_df))
split_sizes = [len(train_df) // n_clients] * n_clients
split_sizes[-1] += len(train_df) - sum(split_sizes)

client_data = {}
start = 0
for i in range(n_clients):
    end = start + split_sizes[i]
    cp = train_df.iloc[train_indices[start:end]]
    pos_rate = cp['Diabetes_binary'].mean()
    client_data[i] = {
        'df': cp,
        'size': len(cp),
        'pos_rate': pos_rate
    }
    print(f"Client {i}: {len(cp)} rows, positive rate: {pos_rate*100:.2f}%")
    start = end

# Step 3: Within each client, split into train/validation (80/20 local)
local_splits = {}
for client_id in range(n_clients):
    cp = client_data[client_id]['df']
    X_local = cp.drop('Diabetes_binary', axis=1)
    y_local = cp['Diabetes_binary']
    
    X_local_train, X_local_val, y_local_train, y_local_val = train_test_split(
        X_local, y_local, test_size=0.25, random_state=42 + client_id, stratify=y_local
    )
    
    local_train_df = pd.concat([X_local_train, y_local_train], axis=1)
    local_val_df = pd.concat([X_local_val, y_local_val], axis=1)
    
    local_splits[client_id] = {
        'train': local_train_df,
        'val': local_val_df,
        'train_size': len(local_train_df),
        'val_size': len(local_val_df)
    }
    
    pos_tr = local_train_df['Diabetes_binary'].mean()
    pos_val = local_val_df['Diabetes_binary'].mean()
    print(f"Client {client_id}: train={local_train_df.shape[0]} rows, val={local_val_df.shape[0]} rows, pos_train={pos_tr*100:.1f}%, pos_val={pos_val*100:.1f}%")

# Step 4: Scale features using fixed-range min-max scaling
# Fit scaler on global train pool only - no client needs another's statistics
continuous_cols = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']

print(f"\n=== Scaling: fitting on global train pool ===")
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_df[continuous_cols])

print(f"Scaler min_: {scaler.min_}")
print(f"Scaler scale_: {scaler.scale_}")
print(f"Feature range after scaling: [0, 1]")

# Transform all sets
for client_id in range(n_clients):
    # Global train pool
    local_splits[client_id]['train'][continuous_cols] = scaler.transform(local_splits[client_id]['train'][continuous_cols])
    # Local validation
    local_splits[client_id]['val'][continuous_cols] = scaler.transform(local_splits[client_id]['val'][continuous_cols])
    
    # Test set (server-side evaluation only)
    test_df[continuous_cols] = scaler.transform(test_df[continuous_cols])

# Verify scaling - check a few values
print(f"\nSample scaled BMI values (first 5): {local_splits[0]['train']['BMI'].head().values}")
print(f"Scaled BMI range: [{local_splits[0]['train']['BMI'].min():.4f}, {local_splits[0]['train']['BMI'].max():.4f}]")

# Save processed splits
import os
os.makedirs(r"C:\Users\patel\fl-diabetes\data\processed", exist_ok=True)

# Save global test set (server-side only)
test_df.to_csv(r"C:\Users\patel\fl-diabetes\data\processed\global_test_set.csv", index=False)
print(f"\nGlobal test set saved: {len(test_df)} rows")

# Save client training data (each client only gets their partition)
for client_id in range(n_clients):
    client_df = local_splits[client_id]['train']
    client_df.to_csv(rf"C:\Users\patel\fl-diabetes\data\partitions\client_{client_id}_train.csv", index=False)

# Save client validation sets
for client_id in range(n_clients):
    client_df = local_splits[client_id]['val']
    client_df.to_csv(rf"C:\Users\patel\fl-diabetes\data\partitions\client_{client_id}_val.csv", index=False)

print(f"\nClient train partitions saved: clients 0-{n_clients-1}")
print(f"Client val partitions saved: clients 0-{n_clients-1}")

# Summary
print(f"""
=== PHASE 2 SUMMARY ===
Global train pool: {len(train_df)} rows (80% of {len(df)} after dedup {253680-24206}=229474)
Global test holdout: {len(test_df)} rows (20%), stratified positive rate: {test_df['Diabetes_binary'].mean()*100:.2f}%

5 clients, each with:
- Local training set: ~27K rows
- Local validation set: ~9K rows (25% of client data, 80/20 local split)
- Fixed-range min-max scaler (sklearn, [0,1]) fitted on global train pool only

Continuous features scaled: {continuous_cols}
Binary features left as-is: 0/1

Key: Test set is server-only, never seen by clients during federation.
FedAvg weighted average uses sample counts n_k / n per round.
""")