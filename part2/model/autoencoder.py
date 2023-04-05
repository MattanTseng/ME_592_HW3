import torch.nn as nn


class AE(nn.Module):

    def __init__(self, encoder, decoder, bottleneck_size=1024):
        super(AE,self).__init__()
        self.encoder = encoder(bottleneck_size)
        self.decoder = decoder(bottleneck_size)
    def forward(self,x ):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
