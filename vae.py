import collections
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
# Define the VAE model with Gaussian output in the decoder
import torch.optim as optim
from PIL.Image import Image
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import torchsummary
from torchvision import transforms
from torchvision.utils import save_image
from livelossplot import PlotLosses
from dataloader import MonetDataset, MonetSubset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 128  # monet images are 256 * 256 pixels, we resize them to 128 * 128


class VAE(nn.Module):
    def __init__(self, input_channels, latent_size):
        super(VAE, self).__init__()

        # Encoder

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 256),  # Adjust the input size based on the image dimensions
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.z_mean = nn.Linear(256, latent_size)
        self.z_log_var = nn.Linear(256, latent_size)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),  # Adjust the output size based on the image dimensions
            nn.ReLU(),
            nn.Linear(512, 4096),  # Adjusted sizes
            nn.ReLU(),
            nn.Unflatten(1, (1024, 2, 2)),  # Reshape to the shape before flattening in the encoder
            nn.BatchNorm2d(1024),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Tanh activation for the output layer to ensure values between -1 and 1
        )

        # self.dec_mean = nn.Linear(128, 128)

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
        # x_mean = self.dec_mean(x_recon)

        return x_recon, mu, log_var


# Loss function for Gaussian VAE
def vae_loss(x, x_mean, mu, log_var, beta=0.1):
    # Reconstruction loss using Gaussian distribution
    recon_loss = 0.5 * torch.sum((x_mean - x) ** 2)

    # KL Divergence term
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return beta * recon_loss + kl_loss


def moving_average(values, window_size):
    i = 0
    moving_averages = []
    while i < len(values) - window_size + 1:
        this_window = values[i: i + window_size]
        window_average = sum(this_window) / window_size
        moving_averages.append(window_average)
        i += 1
    return moving_averages


if __name__ == '__main__':
    # Hyperparameters
    num_epochs = 1000000
    channels = 1
    learning_rate = 0.000001
    weigth_decay = 1e-5
    latent_size = 4096  # Size of the latent space, hyperparameter to optimize

    # LOAD IMAGES

    # Specify the path to your folder containing JPEG images
    data_path = 'data_jpg/monet_jpg'

    # Define transformations
    random_transform = torch.nn.Sequential(
        transforms.Resize((input_size, input_size)),
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Normalize(mean=[0.5], std=[0.5])
    )

    # Initialize datasets
    monet_dataset = MonetDataset(data_path, device, input_size)

    # Initialize data loaders
    monet_loader = DataLoader(monet_dataset, batch_size=len(monet_dataset), shuffle=True, drop_last=True)

    # Initialize Gaussian VAE model
    vae = VAE(channels, latent_size).to(device)
    # vae.weight_init(mean=0, std=0.02) # TODO: check if this is needed
    print("model initialized")

    torchsummary.summary(vae, (1, 128, 128))

    # Optimizer
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate, betas=(0.5, 0.999), weight_decay=weigth_decay)

    # Training loop
    epochs = num_epochs
    z_sample = torch.randn(16, latent_size).to(device)

    # Initialize your liveloss plot
    liveloss = PlotLosses(groups={'loss': ['train loss', 'validation loss', 'train loss MA', 'validation loss MA']})

    # Initialize deque (or another structure) for storing recent losses
    train_losses = collections.deque(maxlen=20)
    val_losses = collections.deque(maxlen=20)

    for epoch in range(epochs):
        logs = {}
        vae.train()
        for batch_idx, data in enumerate(monet_loader):
            optimizer.zero_grad()

            random_transformed_data = random_transform(data)

            x_mean, mu, log_var = vae(random_transformed_data)
            loss = vae_loss(random_transformed_data, x_mean, mu, log_var)

            loss.backward()
            optimizer.step()

            logs['train loss'] = loss.item()
            train_losses.append(logs['train loss'])

            with torch.no_grad():
                vae.eval()

                val_x_mean, val_mu, val_log_var = vae(data)
                val_loss = vae_loss(data, val_x_mean, val_mu, val_log_var)

                logs['validation loss'] = val_loss.item()
                val_losses.append(logs['validation loss'])

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tValidation loss: {:.6f}'.format(
                epoch, epoch, epochs,
                100. * epoch / epochs, loss.item(), val_loss.item()))

            # Calculate moving averages and update logs
            if epoch >= 20:  # Ensure we have enough data for moving average calculation
                logs['train loss MA'] = sum(train_losses) / len(train_losses)
                logs['validation loss MA'] = sum(val_losses) / len(val_losses)
            else:
                logs['train loss MA'] = loss.item()
                logs['validation loss MA'] = val_loss.item()

            if epoch % 100 == 0:
                torch.save(vae.state_dict(), 'vae.pth')

                with torch.no_grad():
                    vae.eval()
                    x_mean, _, _ = vae(data)
                    result_sample = torch.cat([data, x_mean]) * 0.5 + 0.5
                    # result_sample = result_sample.cpu()
                    save_image(result_sample.view(-1, 1, input_size, input_size),
                               'results_rec/sample_' + str(epoch) + '.png')
                    x_mean = vae.decoder(z_sample)
                    result_sample = x_mean * 0.5 + 0.5
                    # result_sample = result_sample.cpu()
                    save_image(result_sample.view(-1, 1, input_size, input_size),
                               'results_gen/sample_' + str(epoch) + '.png')

        liveloss.update(logs)
        if epoch % 100 == 0:
            liveloss.send()

    #
    # vae.load_state_dict(torch.load('vae.pth'))
    # vae.eval()

    # TEST THE VAE - no test dataset created for now
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(monet_loader):
            data = data.to(device)

            x_mean, mu, log_var = vae(data)
            test_loss += vae_loss(data, x_mean, mu, log_var).item()

    test_loss /= len(monet_loader.dataset)
    print('Test set loss: {:.4f}'.format(test_loss))

    # Visualization
    vae.eval()
    n = 10
    painting_size = 128
    with torch.no_grad():
        figure = np.zeros((painting_size * n, painting_size * n))
        for i, xi in enumerate(np.linspace(-2, 2, n)):
            for j, yi in enumerate(np.linspace(-2, 2, n)):
                z_sample = torch.randn(16, latent_size).to(device)
                x_decoded = vae.decoder(z_sample).cpu().numpy()
                digit = x_decoded[0].reshape(painting_size, painting_size)
                figure[
                i * painting_size: (i + 1) * painting_size,
                j * painting_size: (j + 1) * painting_size,
                ] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap="Greys_r")
        plt.show()
