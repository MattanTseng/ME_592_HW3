from torch.utils.data import DataLoader
from torchvision import transforms
from model.autoencoder_conv import AutoEncoderConv
import pytorch_lightning as pl
from utils.data_loading import CombustionSystemDataset
import torch

transform = transforms.Compose([
    transforms.ToTensor()
])


PATH = 'Zare_data/combustion_img_13.mat'

# train_dataset = CombustionSystemDataset(PATH, 'train_set_x', 'train_set_y')

test_dataset = CombustionSystemDataset(PATH, 'test_set_x', 'test_set_y')

train_dataset = CombustionSystemDataset(PATH, 'test_set_x', 'test_set_y' )

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)
gpus = torch.cuda.device_count()

# Train with DDP on multiple gpus. Distributed sampling is also enabled with
# replace_sampler_ddp=True.
trainer = pl.Trainer(
    max_epochs=20,
    gpus=gpus,
    strategy="dp",
    replace_sampler_ddp=True,
)
model = AutoEncoderConv()
trainer.fit(model, train_loader)

trainer.logger._log_graph = True

trainer.test(model, test_loader)

trainer.save_checkpoint('./logging/autoencoder_Conv.ckpt')

