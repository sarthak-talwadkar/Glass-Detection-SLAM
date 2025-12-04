import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class GlassDataset(Dataset):
    """Glass Detection Dataset"""
    
    def __init__(self, root, split='train', img_size=384, augment=False):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.augment = augment
        
        # Get image and mask paths
        self.img_dir = os.path.join(root, split, 'image')
        self.mask_dir = os.path.join(root, split, 'mask')
        
        # Get all image files
        self.images = sorted([f for f in os.listdir(self.img_dir) 
                            if f.endswith(('.jpg', '.png'))])
        
        # Define transforms
        self.img_transform = self._get_img_transform()
        self.mask_transform = self._get_mask_transform()
        
        if augment:
            self.aug_transform = self._get_augmentation_transform()
        else:
            self.aug_transform = None
    
    def _get_img_transform(self):
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _get_mask_transform(self):
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])
    
    def _get_augmentation_transform(self):
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                                  saturation=0.2, hue=0.1)
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image and mask paths
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, 
                               img_name.replace('.jpg', '.png'))
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Apply augmentation if enabled
        if self.aug_transform:
            # Apply same random transform to both image and mask
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            image = self.aug_transform(image)
            torch.manual_seed(seed)
            mask = self.aug_transform(mask)
        
        # Apply transforms
        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        
        # Binarize mask
        mask = (mask > 0.5).float()
        
        return image, mask
