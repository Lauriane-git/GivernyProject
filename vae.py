import matplotlib.pyplot as plt
import numpy as np
import torch
# Define the VAE model with Gaussian output in the decoder
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image


from dataloader import MonetDataset, MonetSubset

input_size = 128 # monet images are 256 * 256 pixels, we resize them to 128 * 128


class VAE(nn.Module):
    def __init__(self, input_channels, latent_size):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(32768, 256),  # Adjust the input size based on the image dimensions
            nn.ReLU(),
        )

        self.z_mean = nn.Linear(256, latent_size)
        self.z_log_var = nn.Linear(256, latent_size)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 32768),  # Adjust the output size based on the image dimensions
            nn.ReLU(),
            nn.Unflatten(1, (128, 16, 16)),  # Reshape to the shape before flattening in the encoder
            # nn.Unflatten(1, (512, 4, 4)),  # Reshape to the shape before flattening in the encoder
            # nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
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
def vae_loss(x, x_mean, mu, log_var):
    # Reconstruction loss using Gaussian distribution
    recon_loss = 0.5 * torch.sum((x_mean - x) ** 2)

    # KL Divergence term
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return 0.01 * recon_loss + kl_loss



if __name__ == '__main__':
    # Hyperparameters
    num_epochs = 1000
    batch_size = 50
    channels = 1
    learning_rate = 0.000005
    weigth_decay = 1e-5
    latent_size = 1024  # Size of the latent space, hyperparameter to optimize

    # LOAD IMAGES

    # Specify the path to your folder containing JPEG images
    data_path = 'data_jpg/monet_jpg'

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),

        # Data augmentation
        # transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally
        # transforms.RandomRotation(degrees=15),  # Random rotation of the image
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # Randomly change brightness, contrast, and saturation

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Initialize datasets
    monet_dataset = MonetDataset(data_path, transform)
    # selected_indices = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    # subset_dataset = MonetSubset(monet_dataset, selected_indices)

    # Initialize data loaders
    monet_loader = DataLoader(monet_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # TRAIN THE MODEL

    # Initialize Gaussian VAE model
    vae = VAE(channels, latent_size).to(device)
    # vae.weight_init(mean=0, std=0.02) # TODO: check if this is needed
    print("model initialized")

    # torchsummary.summary(vae, (1, 128, 128))

    # Optimizer
    optimizer = optim.Adam(vae.parameters(), lr=learning_rate, betas=(0.5, 0.999), weight_decay=weigth_decay)

    # Training loop
    epochs = num_epochs
    z_sample = torch.randn(16, latent_size)

    for epoch in range(epochs):
        vae.train()
        for batch_idx, data in enumerate(monet_loader):

            optimizer.zero_grad()

            x_mean, mu, log_var = vae(data)
            loss = vae_loss(data, x_mean, mu, log_var)

            loss.backward()
            optimizer.step()

            if batch_idx == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, epoch, epochs,
                           100. * epoch / epochs, loss.item()))
                average_sigma = torch.mean(torch.exp(0.5 * log_var).detach()).item()
                print("average sigma: ", average_sigma)
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

    #
    # vae.load_state_dict(torch.load('vae.pth'))
    # vae.eval()

    # TEST THE VAE - no test dataset created for now
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(monet_loader):
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
                z_sample = torch.randn(16, latent_size)
                x_decoded = vae.decoder(z_sample).cpu().numpy()
                digit = x_decoded[0].reshape(painting_size, painting_size)
                figure[
                i * painting_size: (i + 1) * painting_size,
                j * painting_size: (j + 1) * painting_size,
                ] = digit

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap="Greys_r")
        plt.show()
