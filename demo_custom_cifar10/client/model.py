# insert model
import torch
import torch.nn.functional as F
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(self, h1=2048):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(16 * 4 * 4, 10)

        self.dropout = nn.Dropout(p=0.2)

        self.out = nn.LogSoftmax(dim=1)


    def flatten(self, x):
        return x.view(x.size()[0], -1) 


    def forward(self, x):
        x = self.dropout(self.pool(F.relu(self.conv1(x))))
        x = self.dropout(self.pool(F.relu(self.conv2(x))))
        x = self.dropout(self.pool(F.relu(self.conv3(x))))
        x = self.flatten(x)
        x = self.out(x)

        return x