import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

class VGG16Backbone(nn.Module):
    def __init__(self):
        super(VGG16Backbone, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features
        # self.avgpool = vgg16.avgpool
        # self.latent = nn.Linear(512*49, 512*12)

        # Modify the first convolutional layer to accept a single-channel input
        self.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.latent(x)
        return x

class AutoencoderVGG(nn.Module):
    def __init__(self):
        super(AutoencoderVGG, self).__init__()
        self.encoder = VGG16Backbone()

        # Modify the output features of VGG16 to match your input size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),

            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        print("befro" + str(np.shape(x)))
        x = x.view(-1, 512, 7, 3)  # Reshape the tensor to match the expected input shape for the decoder
        print("after" + str(np.shape(x)))

        x = self.decoder(x)
        return x

# Instantiate the autoencoder and move it to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = AutoencoderVGG().to(device)

# test the autoencoder with a random input tensor
input_tensor = torch.rand(1, 1, 250, 100).to(device)
output_tensor = autoencoder(input_tensor)
print(output_tensor.size())  # Should output torch.Size([1, 1, 250, 100])

# Hyperparameters
num_epochs = 100
learning_rate = 1e-3
batch_size = 16

# Create a random dataset for demonstration purposes
data = torch.rand(100, 1, 250, 100)
dataset = torch.utils.data.TensorDataset(data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Set up the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (inputs,) in enumerate(dataloader):
        inputs = inputs.to(device)

        # Forward pass
        outputs = autoencoder(inputs)

        # Calculate the loss
        loss = criterion(outputs, inputs)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

print("Training complete.")