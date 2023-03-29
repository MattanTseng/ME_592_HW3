import torch.nn as nn


class DecoderConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(1024 * 4, 32 * 13 * 32), nn.GELU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, output_padding=1, padding=1, stride=2),  # 32x13 => 8x8
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, output_padding=1, padding=1, stride=2),  # 63x8 => 16x16
            nn.GELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, output_padding=1, padding=1, stride=2),  # 16x16 => 32x32
            nn.Tanh(),  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 32, 13)
        x = self.net(x)
        return x