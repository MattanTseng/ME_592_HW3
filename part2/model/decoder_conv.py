import torch.nn as nn


class DecoderConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(1024, 32 * 13 * 31), nn.GELU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=(5, 3), stride=2, padding=1, output_padding=(1, 0)),
            nn.GELU(),
            nn.ConvTranspose2d(16, 1, kernel_size=(5, 3), stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 31, 13)
        x = self.net(x)
        x = x[:, :, 3:253, 1:101]

        return x