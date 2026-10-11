import torch
import torch.nn as nn
import numpy as np


class LogisticRegression(nn.Module):
    """Logistic regression model: single linear layer + sigmoid."""
    
    def __init__(self, input_dim=21):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        logits = self.linear(x)
        probs = torch.sigmoid(logits)
        return probs


class SmallMLP(nn.Module):
    """Small MLP: 21 -> 64 -> 32 -> 1 (ReLU, optional dropout)."""
    
    def __init__(self, input_dim=21, hidden_dim1=64, hidden_dim2=32, dropout=0.0):
        super(SmallMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.layer3 = nn.Linear(hidden_dim2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x


def get_parameters(model) -> list:
    """Get model parameters as a list of numpy arrays.
    
    Flower uses this format for parameter exchange.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, params: list):
    """Set model parameters from a list of numpy arrays."""
    state_dict = model.state_dict()
    new_params = {k: torch.tensor(v) for k, v in zip(state_dict.keys(), params)}
    state_dict.update(new_params)
    model.load_state_dict(state_dict)