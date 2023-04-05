import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, bottleneck_size ):
        super(Decoder,self).__init__()
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64,out_channels=32, kernel_size=(5,3), stride=2, padding=1, output_padding=(1,0)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32,out_channels=1, kernel_size=(5,3),stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(128,30,12))

        self.dense_layers = nn.Sequential(

            nn.Linear(bottleneck_size,1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128 * 30 * 12),
            nn.ReLU(),
            nn.Dropout(0.2),
           )

    def forward(self, x):
        x = self.dense_layers(x)
        x  = self.unflatten(x)
        x = self.conv_layers(x)
        x = x[:, :, 4:254, 2:102]

        return x


