import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import flwr as fl
from flwr.client import NumPyClient
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# Simple model
model = nn.Linear(21, 1)


# Flower NumPyClient
class DiabetesClient(NumPyClient):
    def get_parameters(self):
        return fl.common.parameters_toNDArrays(model.state_dict())

    def fit(self, parameters, config):
        from src.model import set_parameters
        set_parameters(model, parameters)
        model.train()
        # Dummy data - in reality each client has their partition
        X = np.random.rand(100, 21).astype(np.float32)
        y = np.random.randint(0, 2, 100)
        dataset = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=16)

        optimizer = torch.optim.SGD(model.parameters(), lr=config.get("lr", 0.01))
        criterion = nn.BCEWithLogitsLoss()

        for _ in range(1):  # 1 local epoch
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()

        return fl.common.parameters_toNDArrays(model.state_dict()), len(dataset), {"accuracy": 0.5}

    def evaluate(self, parameters, config):
        from src.model import set_parameters
        set_parameters(model, parameters)
        model.eval()
        X = np.random.rand(20, 21).astype(np.float32)
        y = np.random.randint(0, 2, 20)
        dataset = TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=16)

        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                loss = criterion(model(batch_x), batch_y.unsqueeze(1))
                total_loss += loss.item()
                total += batch_y.size(0)
        return fl.common.scalar_to_tcalar(0.0), len(dataset), {"loss": total_loss / len(loader)}


# Start Flower server strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_available_clients=2,
)

# Start server
server_config = fl.server.ServerConfig(num_rounds=1)

# Use simulation
fl.simulation.start_simulation(
    client_fn=lambda cid: DiabetesClient(),
    num_clients=2,
    strategy=strategy,
    config=fl.simulation.ServerConfig(num_rounds=1),
    repeat=False,
)
print("Flower simulation completed")