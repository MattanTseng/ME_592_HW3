import torch.nn as nn
import torch.nn.functional as F
import torch


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(64, 512)
        self.fc2 = nn.Linear(512, 250*100)

    def forward(self, x):
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.reshape(x, (-1, 1, 250, 100))
        return x
