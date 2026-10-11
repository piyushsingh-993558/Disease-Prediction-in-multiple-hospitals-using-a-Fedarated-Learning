import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

from model import LogisticRegression, get_parameters, set_parameters
from train_eval import train_one_epoch, evaluate, create_dataloader


def run_centralized_baseline(train_df, test_df, model_type='logistic', epochs=10, lr=0.01, batch_size=64, device='cpu'):
    """Train the same model on all pooled training data (centralized upper bound).
    
    Returns dict with metrics on the global test set.
    """
    print(f"=== Centralized Baseline: {model_type.upper()} ===")
    
    # Prepare data
    continuous_cols = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']
    X_train = train_df.drop('Diabetes_binary', axis=1)
    y_train = train_df['Diabetes_binary']
    X_test = test_df.drop('Diabetes_binary', axis=1)
    y_test = test_df['Diabetes_binary']
    
    # Scale features using same scaler approach
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[continuous_cols])
    
    X_train[continuous_cols] = scaler.transform(X_train[continuous_cols])
    X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])
    
    # Compute pos_weight for imbalance handling
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"Class counts - Pos: {n_pos}, Neg: {n_neg}, pos_weight: {pos_weight:.2f}")
    
    # Initialize model
    if model_type == 'logistic':
        model = LogisticRegression(input_dim=21)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    else:
        model = SmallMLP(input_dim=21)
        criterion = nn.BCEWithLogitsLoss()
    
    model = model.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    # Train on full pooled data
    train_loader = create_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True, device=device)
    
    print(f"Training {model_type} for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        avg_loss, acc, prec, rec, f1, auc = train_one_epoch(
            model, train_loader, optimizer, criterion, device=device
        )
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch}: loss={avg_loss:.4f}, acc={acc:.4f}, recall={rec:.4f}, f1={f1:.4f}, auc={auc:.4f}")
    
    # Evaluate on global test set
    test_loader = create_dataloader(X_test, y_test, batch_size=batch_size, shuffle=False, device=device)
    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = evaluate(model, test_loader, criterion, device=device)
    
    print(f"\nCentralized {model_type} results on global test set:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall: {test_rec:.4f}")
    print(f"  F1: {test_f1:.4f}")
    print(f"  AUC-ROC: {test_auc:.4f}")
    
    return {
        'model': model,
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'test_f1': test_f1,
        'test_auc': test_auc,
        'pos_weight': pos_weight
    }


if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv(r"C:\Users\patel\fl-diabetes\data\processed\diabetes_cleaned.csv")
    test_df = pd.read_csv(r"C:\Users\patel\fl-diabetes\data\processed\global_test_set.csv")
    
    # Run centralized baseline with logistic regression
    results = run_centralized_baseline(
        train_df, test_df, 
        model_type='logistic', 
        epochs=20, 
        lr=0.01, 
        batch_size=64
    )
    
    print(f"\n=== CENTRALIZED BASELINE SUMMARY ===")
    for key, val in results.items():
        if key != 'model':
            print(f"  {key}: {val:.4f}")