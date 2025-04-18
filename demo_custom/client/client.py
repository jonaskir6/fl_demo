from typing import OrderedDict
import flwr as fl
import numpy as np
import train
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sys
import model as ml_model
from collections import Counter
from torchvision import datasets, transforms
import json

# Load configuration
with open('config.json', 'r') as file:
    config = json.load(file)

num_epochs = config['num_epochs']  
batch_size = config['batch_size']


train_data = None
test_data = None

# ----------------------------------------




# Add code to load your data into train_data and test_data




# ----------------------------------------

model = ml_model.Model()

train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=False)


class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]
    
    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    # returns weights of MNIST netowrk after training
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train.train(model, train_dl, test_dl, 10)
        return self.get_parameters(config={}), len(train_data), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc = train.evaluate(model, test_dl)
        return 0.0, 10, {"accuracy: ": acc}

fl.client.start_client(
    server_address="server:5002", 
    client=FlowerClient().to_client(),
    grpc_max_message_length=2 * 1024 * 1024 *1024 -1
)