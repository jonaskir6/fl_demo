from typing import OrderedDict
import flwr as fl
from torchvision import datasets, transforms
import numpy as np
import train
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import sys
import model
from collections import Counter

    
def getDist(y):
    counts = Counter(y)
    classes = list(counts.keys())
    class_counts = list(counts.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(classes, class_counts, color='skyblue') 
    
    plt.title("Count of data classes")
    plt.xlabel("Classes")
    plt.ylabel("Count")

    plt.savefig("client_one_count.png")
    
# Get data based on distribution
def getData(dist, x, y):
    dx = []
    dy = []
    counts = [0 for i in range(10)]
    for i in range(len(x)):
        if counts[y[i]]<dist[y[i]]:
            dx.append(x[i])
            dy.append(y[i])
            counts[y[i]] += 1
        
    return np.array(dx), np.array(dy)

model = model.Linear()

transform = transforms.Compose([transforms.ToTensor(), 
                torch.nn.Flatten(0)])

train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, transform=transform)


# ### Convert to numpy arrays for getData function ###
train_x_data = np.array([train_data[i][0].numpy() for i in range(len(train_data))])
train_y_data = np.array([train_data[i][1] for i in range(len(train_data))])

dist = torch.randint(100,5001, (10,)).tolist()

#dist = [4000, 10, 3000, 30, 4500, 70, 3500, 20, 5000, 10]


train_x, train_y = getData(dist, train_x_data, train_y_data)

getDist(train_y)
# ### Convert to numpy arrays for getData function ###

filtered_train_x_tensor = torch.tensor(train_x).float()
filtered_train_y_tensor = torch.tensor(train_y).long()

filtered_train_dataset = TensorDataset(filtered_train_x_tensor, filtered_train_y_tensor)

train_dl = DataLoader(filtered_train_dataset, batch_size=128, shuffle=True)
test_dl = DataLoader(test_data, batch_size=128, shuffle=False)


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
        train.train_model(model, train_dl, 1, 'samples.png')
        return self.get_parameters(config={}), len(train_x), {}
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = train.evaluate(test_dl, model)
        return float(loss), 10, {"accuracy:": 0.0}

fl.client.start_client(
        server_address="localhost:"+str(sys.argv[1]), 
        client=FlowerClient().to_client(),
        grpc_max_message_length=2 * 1024 * 1024 *1024 -1
)