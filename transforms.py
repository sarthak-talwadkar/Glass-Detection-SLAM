# utils/transforms.py
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF

class RandomRotation90:
    """Randomly rotate by 0, 90, 180, or 270 degrees"""
    def __call__(self, img, mask):
        angle = random.choice([0, 90, 180, 270])
        return TF.rotate(img, angle), TF.rotate(mask, angle)

class RandomCrop:
    """Random crop with same crop for image and mask"""
    def __init__(self, size):
        self.size = size
    
    def __call__(self, img, mask):
        i, j, h, w = transforms.RandomCrop.get_params(
            img, output_size=(self.size, self.size))
        img = TF.crop(img, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        return img, mask