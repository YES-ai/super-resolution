import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)

class ImageDataset(Dataset):
  def __init__(self, root, normalize=True):
    super(ImageDataset, self).__init__()
    self.files = sorted(glob.glob(root + "/*.*"))
    self.normalize = normalize

  def __getitem__(self, index):
    img = Image.open(self.files[index % len(self.files)])
    hr_height, hr_width = img.size
    # Transforms for low resolution images and high resolution images
    lr_transform = []
    lr_transform.append(transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC))
    lr_transform.append(transforms.ToTensor())
    if self.normalize:
      lr_transform.append(transforms.Normalize(mean, std))
    lr_transform = transforms.Compose(lr_transform)

    hr_transform = []
    hr_transform.append(transforms.Resize((hr_height, hr_height), Image.BICUBIC))
    hr_transform.append(transforms.ToTensor())
    if self.normalize:
      hr_transform.append(transforms.Normalize(mean, std))
    hr_transform = transforms.Compose(hr_transform)

    img_lr = lr_transform(img)
    img_hr = hr_transform(img)
    return img_lr, img_hr

  def __len__(self):
    return len(self.files)

class Set14(ImageDataset):
  def __init__(self, normalize):
    super(Set14, self).__init__("./data/set14/HR/", normalize=normalize)

