# from https://www.kaggle.com/code/anthonytherrien/artistic-ai-style-transfer


import os

from PIL import Image
from torch.utils.data import Dataset

color_space = 'RGB' # color
color_space = 'L' # grayscale, comment out if you want to use a color image
color_channel_map = {'RGB': 3, 'L': 1} # number of channels for RGB or grayscale images
color_channels = color_channel_map[color_space] # number of channels for the color space

original_image_size = 256 # size of the original image

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
        image = Image.open(image_path).convert(color_space) 
        if self.transform:
            image = self.transform(image)
        return image

def image_cutter_fun(image, w_size, h_size):
    w, h = image.size
    w_num = w // w_size
    h_num = h // h_size
    images = []
    for i in range(w_num):
        for j in range(h_num):
            images.append(image.crop((i*w_size, j*h_size, (i+1)*w_size, (j+1)*h_size)))
    return images

def image_cutter(w_size, h_size):
    return lambda image: image_cutter_fun(image, w_size, h_size)

def image_cutter_parameters(xbyx, w_before=original_image_size, h_before=original_image_size):
    return w_before // xbyx, h_before // xbyx

def image_resize(w_size, h_size):
    return lambda image: image.resize((w_size, h_size))

# create a dataset of small images from a folder of images,
# using a function to cut the images into small pieces:
#   - image_cutter: cut the image into small pieces of size w_size x h_size
#   - image_resize: resize the image to size w_size x h_size
class SmallImageDataset(Dataset):
    def __init__(self, folder_path, cutter):
        self.folder_path = folder_path
        self.image_path_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                       f.endswith(('.jpg', '.jpeg', '.png'))]
        images = []
        for path in self.image_path_list:
            image = Image.open(path).convert(color_space)
            images.extend(cutter(image))
        self.images = images
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]
    
