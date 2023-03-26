import torch.nn as nn
import torch.nn.functional as F
import torch


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(250 * 100, 512)
        self.fc2 = nn.Linear(512, 64)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
