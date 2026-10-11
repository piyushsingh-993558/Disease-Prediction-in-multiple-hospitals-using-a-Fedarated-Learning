import pandas as pd

# Verify global test set
test = pd.read_csv(r"C:\Users\patel\fl-diabetes\data\processed\global_test_set.csv")
print(f"Global test set: {test.shape}")
print(f"Positive rate: {test['Diabetes_binary'].mean()*100:.2f}%")

# Verify a client train partition
train0 = pd.read_csv(r"C:\Users\patel\fl-diabetes\data\partitions\client_0_train.csv")
print(f"\nClient 0 train: {train0.shape}")
print(f"Positive rate: {train0['Diabetes_binary'].mean()*100:.2f}%")

# Verify client val partition
val0 = pd.read_csv(r"C:\Users\patel\fl-diabetes\data\partitions\client_0_val.csv")
print(f"\nClient 0 val: {val0.shape}")
print(f"Positive rate: {val0['Diabetes_binary'].mean()*100:.2f}%")

# Quick summary of all partitions
print("\n=== All partitions ===")
for i in range(5):
    t = pd.read_csv(f"C:\\Users\\patel\\fl-diabetes\\data\\partitions\\client_{i}_train.csv")
    v = pd.read_csv(f"C:\\Users\\patel\\fl-diabetes\\data\\partitions\\client_{i}_val.csv")
    print(f"Client {i}: train={t.shape[0]}, val={v.shape[0]}, train_pos={t['Diabetes_binary'].mean()*100:.1f}%, val_pos={v['Diabetes_binary'].mean()*100:.1f}%")