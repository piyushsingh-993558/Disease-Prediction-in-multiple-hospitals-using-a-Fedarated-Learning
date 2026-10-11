import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from model import LogisticRegression
from train_eval import create_dataloader, evaluate, train_one_epoch


def run_local_only_baseline(partitions_dir, global_test_set, n_clients=5, batch_size=64, device='cpu'):
    """Local-only baseline: each hospital trains alone on only its own data.
    
    Returns per-client metrics and average.
    """
    print(f"=== Local-Only Baseline: {n_clients} clients ===")
    
    # Load global test set for fair evaluation
    test_df = pd.read_csv(global_test_set)
    continuous_cols = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    test_X = test_df.drop('Diabetes_binary', axis=1)
    test_y = test_df['Diabetes_binary']
    test_X[continuous_cols] = scaler.fit_transform(test_X[continuous_cols])
    test_loader = create_dataloader(test_X, test_y, batch_size=batch_size, shuffle=False, device=device)
    
    # Evaluate global model on global test set (or could load a centralized model)
    # For local-only baseline, we evaluate each client's local model on its own data
    # and also on the global test set for comparison
    
    client_results = {}
    all_client_recalls = []
    all_client_f1s = []
    all_client_aucs = []
    
    for client_id in range(n_clients):
        # Load client's local training data
        train_path = f"{partitions_dir}/client_{client_id}_train.csv"
        val_path = f"{partitions_dir}/client_{client_id}_val.csv"
        
        try:
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
        except FileNotFoundError:
            print(f"Client {client_id} data not found, skipping...")
            continue
        
        # Scale features using sklearn scaler fit on this client's train data
        train_X = train_df.drop('Diabetes_binary', axis=1)
        train_y = train_df['Diabetes_binary']
        val_X = val_df.drop('Diabetes_binary', axis=1)
        val_y = val_df['Diabetes_binary']
        
        scaler_c = MinMaxScaler(feature_range=(0, 1))
        scaler_c.fit(train_X[continuous_cols])
        
        train_X[continuous_cols] = scaler_c.transform(train_X[continuous_cols])
        val_X[continuous_cols] = scaler_c.transform(val_X[continuous_cols])
        
        # Train model on this client's data only
        model = LogisticRegression(input_dim=21)
        model = model.to(device)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.BCEWithLogitsLoss()
        
        train_loader = create_dataloader(train_X, train_y, batch_size=batch_size, shuffle=True, device=device)
        
        print(f"Training client {client_id} locally for 20 epochs...")
        for epoch in range(1, 21):
            avg_loss, acc, prec, rec, f1, auc = train_one_epoch(
                model, train_loader, optimizer, criterion, device=device
            )
        
        # Evaluate on local validation set
        val_loader = create_dataloader(val_X, val_y, batch_size=batch_size, shuffle=False, device=device)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate(model, val_loader, criterion, device=device)
        
        # Evaluate on global test set
        test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = evaluate(model, test_loader, criterion, device=device)
        
        client_results[client_id] = {
            'val_loss': val_loss, 'val_accuracy': val_acc, 'val_precision': val_prec,
            'val_recall': val_rec, 'val_f1': val_f1, 'val_auc': val_auc,
            'test_loss': test_loss, 'test_accuracy': test_acc, 'test_precision': test_prec,
            'test_recall': test_rec, 'test_f1': test_f1, 'test_auc': test_auc
        }
        
        all_client_recalls.append(test_rec)
        all_client_f1s.append(test_f1)
        all_client_aucs.append(test_auc)
        
        pos_rate = train_y.mean()
        print(f"Client {client_id}: train_pos={pos_rate*100:.1f}%, "
              f"val_recall={val_rec:.4f}, test_recall={test_rec:.4f}, "
              f"test_auc={test_auc:.4f}")
    
    # Summary
    if client_results:
        avg_recall = np.mean(all_client_recalls)
        avg_f1 = np.mean(all_client_f1s)
        avg_auc = np.mean(all_client_aucs)
        
        print(f"\n=== Local-Only Baseline Summary ===")
        print(f"Average test recall: {avg_recall:.4f}")
        print(f"Average test F1: {avg_f1:.4f}")
        print(f"Average test AUC-ROC: {avg_auc:.4f}")
        
        for cid, results in client_results.items():
            print(f"  Client {cid}: test_recall={results['test_recall']:.4f}, "
                  f"test_f1={results['test_f1']:.4f}, test_auc={results['test_auc']:.4f}")
    
    return client_results


if __name__ == "__main__":
    results = run_local_only_baseline(
        partitions_dir=r"C:\Users\patel\fl-diabetes\data\partitions",
        global_test_set=r"C:\Users\patel\fl-diabetes\data\processed\global_test_set.csv",
        n_clients=5,
        batch_size=64
    )