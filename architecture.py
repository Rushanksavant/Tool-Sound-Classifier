import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=11, kernel_size=5) # Convolution layer 1
        self.conv2 = nn.Conv2d(in_channels=11, out_channels=27, kernel_size=5) # Convolution layer 2
        
        self.fc1 = nn.Linear(in_features=27 * 29 * 29, out_features=100) # Linear layer 1 [Linear layer also called Fully Cnnected Layer]
        self.out = nn.Linear(in_features=100, out_features=5) # Linear layer 2 (output layer)
        
    def forward(self, t):
        # input layer
        t=t
        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        # (4) hidden linear layer
        t = t.reshape(-1, 27 * 29 * 29)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) output layer
        t = self.out(t)
        
        return t