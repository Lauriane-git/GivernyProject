import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import MonetDataset


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


if __name__ == '__main__':
    # Hyperparameters
    IMAGE_DIM = 256
    num_epochs = 512
    batch_size = 16
    learning_rate = 1e-4
    weigth_decay = 1e-3
    input_size = 128 * 128  # monet images are 256 * 256 pixels, we resize them to 128 * 128
    latent_size = 40  # Size of the latent space, hyperparameter to optimize

    # LOAD IMAGES

    # Specify the path to your folder containing JPEG images
    data_path = 'data_jpg/monet_jpg'

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),

        # Data augmentation
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally
        transforms.RandomRotation(degrees=15),  # Random rotation of the image
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # Randomly change brightness, contrast, and saturation

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Initialize datasets
    monet_dataset = MonetDataset(data_path, transform)

    # Initialize data loaders
    monet_loader = DataLoader(monet_dataset, batch_size=batch_size, shuffle=True)

    # TRAIN THE MODEL

    # Initialize Gaussian VAE model
    vae = VAE(input_size, latent_size).to(device)
    print("model initialized")

    # Optimizer
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    # Training loop
    epochs = num_epochs

    for epoch in range(epochs):
        vae.train()
        for batch_idx, data in enumerate(monet_loader):
            data = data.view(-1, input_size)
            optimizer.zero_grad()

            x_mean, x_log_var, mu, log_var = vae(data)
            loss = vae_loss(data, x_mean, x_log_var, mu, log_var)

            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(monet_loader.dataset),
                           100. * batch_idx / len(monet_loader), loss.item()))
                average_sigma = torch.mean(torch.exp(0.5 * log_var).detach()).item()
                print("average sigma: ", average_sigma)
            torch.save(vae.state_dict(), 'vae.pth')

    vae.load_state_dict(torch.load('vae.pth'))
    vae.eval()

    # # TEST THE VAE
    # vae.eval()
    # test_loss = 0
    # with torch.no_grad():
    #     for i, (data, _) in enumerate(monet_loader):
    #         data = data.view(-1, input_size)
    #         x_mean, x_log_var, mu, log_var = vae(data)
    #         test_loss += vae_loss(data, x_mean, x_log_var, mu, log_var).item()
    #
    # test_loss /= len(monet_loader.dataset)
    # print('Test set loss: {:.4f}'.format(test_loss))

    # Visualization
    vae.eval()
    n = 1
    painting_size = 128
    with torch.no_grad():
        figure = np.zeros((painting_size * n, painting_size * n))
        for i, xi in enumerate(np.linspace(-2, 2, n)):
            for j, yi in enumerate(np.linspace(-2, 2, n)):
                z_sample = torch.randn(1, latent_size)
                x_decoded = vae.dec_mean(vae.decoder(z_sample)).cpu().numpy()
                digit = x_decoded[0].reshape(painting_size, painting_size)
                figure[
                i * painting_size: (i + 1) * painting_size,
                j * painting_size: (j + 1) * painting_size,
                ] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap="Greys_r")
        plt.show()
