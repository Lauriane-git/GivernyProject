# from https://www.kaggle.com/code/anthonytherrien/artistic-ai-style-transfer


import os

from PIL import Image
from torch.utils.data import Dataset

color_space = 'RGB' # color
color_space = 'L' # grayscale, comment out if you want to use a color image
color_channel_map = {'RGB': 3, 'L': 1} # number of channels for RGB or grayscale images
color_channels = color_channel_map[color_space] # number of channels for the color space

class MonetDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                       f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert(color_space) 
        if self.transform:
            image = self.transform(image)
        return image


class PhotoDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                       f.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert(color_space) 
        if self.transform:
            image = self.transform(image)
        return image
