import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class IoULoss(nn.Module):
    """Intersection over Union Loss"""
    
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        # Calculate IoU
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Return loss (1 - IoU)
        return 1 - iou


class BoundaryLoss(nn.Module):
    """Boundary-aware loss for edge refinement"""
    
    def __init__(self, kernel_size=5):
        super(BoundaryLoss, self).__init__()
        self.kernel_size = kernel_size
        
    def get_boundary(self, mask):
        """Extract boundary from mask using morphological operations"""
        kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size).cuda()
        
        # Dilate and erode to get boundary
        dilated = F.conv2d(mask, kernel, padding=self.kernel_size//2)
        dilated = (dilated > 0).float()
        
        eroded = F.conv2d(mask, kernel, padding=self.kernel_size//2)
        eroded = (eroded == self.kernel_size * self.kernel_size).float()
        
        # Boundary = dilated - eroded
        boundary = dilated - eroded
        
        return boundary
    
    def forward(self, pred, target):
        # Get boundaries
        target_boundary = self.get_boundary(target)
        
        # Apply sigmoid to prediction
        pred_sigmoid = torch.sigmoid(pred)
        
        # Calculate boundary loss (weighted BCE)
        boundary_weight = target_boundary * 4 + 1  # Weight boundaries more
        bce = F.binary_cross_entropy(pred_sigmoid, target, reduction='none')
        weighted_bce = bce * boundary_weight
        
        return weighted_bce.mean()


class GlassDetectionLoss(nn.Module):
    """Combined loss for glass detection"""
    
    def __init__(self, bce_weight=1.0, iou_weight=0.5, boundary_weight=0.5):
        super(GlassDetectionLoss, self).__init__()
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.boundary_weight = boundary_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.iou_loss = IoULoss()
        self.boundary_loss = BoundaryLoss()
    
    def forward(self, pred, target):
        # Calculate individual losses
        bce = self.bce_loss(pred, target)
        iou = self.iou_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        
        # Combine losses
        total_loss = (self.bce_weight * bce + 
                     self.iou_weight * iou + 
                     self.boundary_weight * boundary)
        
        return {
            'total': total_loss,
            'bce': bce,
            'iou': iou,
            'boundary': boundary
        }
