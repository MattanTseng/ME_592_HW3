from .encoder_conv import EncoderConv
from .decoder_conv import DecoderConv
import torch.nn.functional as F
import pytorch_lightning as pl
import torch


class AutoEncoderConv(pl.LightningModule):

    def __init__(self, encoder=EncoderConv, decoder=DecoderConv):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder()
        self.decoder = decoder()
        self.example_input_array = torch.zeros(2, 1, 250, 100)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        # we don't need the label
        x, _ = batch
        x_hat = self.forward(x)


        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat)
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
