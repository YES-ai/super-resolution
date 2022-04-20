import glob
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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
  def __init__(self, root, normalize=True, target_size=(512,512)):
    super(ImageDataset, self).__init__()
    self.files = sorted(glob.glob(root + "/*.*"))
    self.normalize = normalize
    self.target_size = target_size

  def __getitem__(self, index):
    img = Image.open(self.files[index % len(self.files)])

    img = transforms.RandomResizedCrop(self.target_size)(img)
    hr_height, hr_width = self.target_size

    # Transforms for low resolution images and high resolution images
    lr_transform = []
    lr_transform.append(transforms.Resize((hr_height // 2, hr_width // 2), transforms.InterpolationMode.BICUBIC))
    lr_transform.append(transforms.ToTensor())
    if self.normalize:
      lr_transform.append(transforms.Normalize(mean, std))
    lr_transform = transforms.Compose(lr_transform)

    hr_transform = []
    hr_transform.append(transforms.Resize((hr_height, hr_width), transforms.InterpolationMode.BICUBIC))
    hr_transform.append(transforms.ToTensor())
    if self.normalize:
      hr_transform.append(transforms.Normalize(mean, std))
    hr_transform = transforms.Compose(hr_transform)

    ur_transform = []
    ur_transform.append(transforms.Resize((hr_height // 8, hr_width // 8), transforms.InterpolationMode.BICUBIC))
    ur_transform.append(transforms.Resize((hr_height, hr_width), transforms.InterpolationMode.BICUBIC))
    ur_transform.append(transforms.ToTensor())
    if self.normalize:
      ur_transform.append(transforms.Normalize(mean, std))
    ur_transform = transforms.Compose(ur_transform)

    img_lr = lr_transform(img)
    img_hr = hr_transform(img)
    img_ur = ur_transform(img)
    return img_lr.permute(1,2,0).numpy(), img_hr.permute(1,2,0).numpy(), img_ur.permute(1,2,0).numpy()

  def __len__(self):
    return len(self.files)
