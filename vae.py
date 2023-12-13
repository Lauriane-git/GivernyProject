import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


# Define the VAE model with Gaussian output in the decoder
class VAE(nn.Module):
    def __init__(self, input_size, latent_size):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.z_mean = nn.Linear(128, latent_size)
        self.z_log_var = nn.Linear(128, latent_size)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.Tanh(),  # Tanh activation for the output layer to ensure values between -1 and 1
        )

        self.dec_mean = nn.Linear(input_size, input_size)
        self.dec_log_var = nn.Linear(input_size, input_size)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        mu = self.z_mean(x)
        log_var = self.z_log_var(x)

        # Reparameterization trick
        z = self.reparameterize(mu, log_var)

        # Decode
        x_recon = self.decoder(z)
        x_mean = self.dec_mean(x_recon)
        x_log_var = self.dec_log_var(x_recon)

        return x_mean, x_log_var, mu, log_var


# Loss function for Gaussian VAE
def vae_loss(x, x_mean, x_log_var, mu, log_var):
    # Reconstruction loss using Gaussian distribution
    recon_loss = 0.5 * torch.sum(((x_mean - x) ** 2 / torch.exp(x_log_var) + x_log_var))

    # KL Divergence term
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + kl_loss

#LOAD IMAGES
# Define the transformation you want to apply to the images
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the images to the input size
    transforms.ToTensor()
])

# Specify the path to your folder containing JPEG images
data_path = 'data_jpg/'

# Create a dataset using ImageFolder
custom_dataset = ImageFolder(root=data_path, transform=transform)

# Create a DataLoader for your custom dataset
batch_size = 128
shuffle = True
custom_data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=shuffle)

# Initialize Gaussian VAE model
input_size = 256 * 256  # monet images are 256 * 256 pixels
latent_size = 40  # Size of the latent space, hyperparameter to optimize
vae = VAE(input_size, latent_size)

# Optimizer
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# Training loop
epochs = 500
for epoch in range(epochs):
    vae.train()
    for batch_idx, (data, _) in enumerate(custom_data_loader):
        data = data.view(-1, input_size)
        optimizer.zero_grad()

        x_mean, x_log_var, mu, log_var = vae(data)
        loss = vae_loss(data, x_mean, x_log_var, mu, log_var)

        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(custom_data_loader.dataset),
                       100. * batch_idx / len(custom_data_loader), loss.item()))
            average_sigma = torch.mean(torch.exp(0.5 * log_var).detach()).item()
            print("average sigma: ",average_sigma)

# Testing the VAE
vae.eval()
test_loss = 0
with torch.no_grad():
    for i, (data, _) in enumerate(custom_data_loader):
        data = data.view(-1, input_size)
        x_mean, x_log_var, mu, log_var = vae(data)
        test_loss += vae_loss(data, x_mean, x_log_var, mu, log_var).item()

test_loss /= len(custom_data_loader.dataset)
print('Test set loss: {:.4f}'.format(test_loss))

# Visualization
vae.eval()
with torch.no_grad():
    n = 15
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    for i, xi in enumerate(np.linspace(-2, 2, n)):
        for j, yi in enumerate(np.linspace(-2, 2, n)):
            z_sample = torch.randn(1, latent_size)
            x_decoded = vae.dec_mean(vae.decoder(z_sample)).cpu().numpy()
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
            i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap="Greys_r")
    plt.show()
