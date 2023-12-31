# from https://www.kaggle.com/code/anthonytherrien/artistic-ai-style-transfer


import os

from PIL import Image
from torch.utils.data import Dataset


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
        image = Image.open(image_path).convert('L')  # could convert to "RGB" if you want to use a color image
        if self.transform:
            image = self.transform(image)
        return image


class MonetSubset(Dataset):
    def __init__(self, base_dataset, indices, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        sample = self.base_dataset[base_idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


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
        image = Image.open(image_path).convert('L')  # could convert to "RGB" if you want to use a color image
        if self.transform:
            image = self.transform(image)
        return image
