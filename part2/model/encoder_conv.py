import torch.nn as nn
import torch


class EncoderConv(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 3), stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=(5, 3), stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU()
        )

        self.fc1 = nn.Linear(32 * 13 * 31, 1024*4)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

