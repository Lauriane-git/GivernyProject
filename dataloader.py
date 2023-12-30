# from https://www.kaggle.com/code/anthonytherrien/artistic-ai-style-transfer


import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MonetDataset(Dataset):
    def __init__(self, folder_path, device, input_size):
        self.folder_path = folder_path
        self.images = []  # store the images here
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        for image_file in self.image_files:
            image_path = os.path.join(folder_path, image_file)
            image = Image.open(image_path).convert('L')
            image = transforms.ToTensor()(image).to(device)
            image = torch.nn.Sequential(
                transforms.Resize((input_size, input_size)),
                transforms.Normalize(mean=[0.5], std=[0.5])
            )(image)
            self.images.append(image)  # store the PIL image, not the tensor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Fetch the image from memory
        image = self.images[idx % len(self.images)]

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
