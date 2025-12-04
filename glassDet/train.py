"""
Training script for GDNet
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime

from models.GDNet import GDNet
from utils.dataset import GlassDataset
from utils.loss import GlassDetectionLoss
from utils.metrics import calculate_metrics


def parse_args():
    parser = argparse.ArgumentParser(description='Train GDNet')
    
    # Dataset
    parser.add_argument('--data-root', type=str, default='./data',
                        help='Path to GDD dataset')
    parser.add_argument('--img-size', type=int, default=384,
                        help='Input image size')
    
    # Model
    parser.add_argument('--backbone', type=str, default='resnext101_32x8d',
                        choices=['resnext101_32x8d', 'resnet101'],
                        help='Backbone architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained backbone')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Training
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Loss weights
    parser.add_argument('--bce-weight', type=float, default=1.0,
                        help='Weight for BCE loss')
    parser.add_argument('--iou-weight', type=float, default=0.5,
                        help='Weight for IoU loss')
    parser.add_argument('--boundary-weight', type=float, default=0.5,
                        help='Weight for boundary loss')
    
    # Logging
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Logging interval (iterations)')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Validation interval (epochs)')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Model saving interval (epochs)')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    
    # GPU
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU id to use')
    
    return parser.parse_args()


def train_epoch(model, dataloader, criterion, optimizer, epoch, args, writer):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    running_bce = 0.0
    running_iou = 0.0
    running_boundary = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.epochs}')
    
    for i, (images, masks) in enumerate(pbar):
        images = images.cuda()
        masks = masks.cuda()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss_dict = criterion(outputs, masks)
        loss = loss_dict['total']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update statistics
        running_loss += loss.item()
        running_bce += loss_dict['bce'].item()
        running_iou += loss_dict['iou'].item()
        running_boundary += loss_dict['boundary'].item()
        
        # Log to tensorboard
        if i % args.log_interval == 0:
            global_step = epoch * len(dataloader) + i
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/BCE', loss_dict['bce'].item(), global_step)
            writer.add_scalar('Train/IoU', loss_dict['iou'].item(), global_step)
            writer.add_scalar('Train/Boundary', loss_dict['boundary'].item(), global_step)
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'BCE': f'{loss_dict["bce"].item():.4f}',
            'IoU': f'{loss_dict["iou"].item():.4f}'
        })
    
    # Return average losses
    n_batches = len(dataloader)
    return {
        'loss': running_loss / n_batches,
        'bce': running_bce / n_batches,
        'iou': running_iou / n_batches,
        'boundary': running_boundary / n_batches
    }


def validate(model, dataloader, criterion, epoch, args):
    """Validate the model"""
    model.eval()
    
    running_loss = 0.0
    all_preds = []
    all_masks = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        
        for images, masks in pbar:
            images = images.cuda()
            masks = masks.cuda()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss_dict = criterion(outputs, masks)
            running_loss += loss_dict['total'].item()
            
            # Store predictions and masks for metrics
            preds = torch.sigmoid(outputs) > 0.5
            all_preds.append(preds.cpu().numpy())
            all_masks.append(masks.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    metrics = calculate_metrics(all_preds, all_masks)
    
    # Add average loss to metrics
    metrics['loss'] = running_loss / len(dataloader)
    
    return metrics


def save_checkpoint(model, optimizer, epoch, metrics, args):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'args': args
    }
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        args.checkpoint_dir, 
        f'gdnet_epoch_{epoch}.pth'
    )
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved: {checkpoint_path}')
    
    # Save best model
    if not hasattr(save_checkpoint, 'best_iou') or metrics['iou'] > save_checkpoint.best_iou:
        save_checkpoint.best_iou = metrics['iou']
        best_path = os.path.join(args.checkpoint_dir, 'gdnet_best.pth')
        torch.save(checkpoint, best_path)
        print(f'Best model saved: {best_path}')


def main():
    args = parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize tensorboard
    log_dir = os.path.join('runs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    
    # Create model
    model = GDNet(backbone=args.backbone, pretrained=args.pretrained)
    model = model.cuda()
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Load checkpoint if resuming
    start_epoch = 1
    if args.resume:
        print(f'Loading checkpoint from {args.resume}')
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Create datasets and dataloaders
    train_dataset = GlassDataset(
        root=args.data_root,
        split='train',
        img_size=args.img_size,
        augment=True
    )
    
    val_dataset = GlassDataset(
        root=args.data_root,
        split='val',
        img_size=args.img_size,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create loss function
    criterion = GlassDetectionLoss(
        bce_weight=args.bce_weight,
        iou_weight=args.iou_weight,
        boundary_weight=args.boundary_weight
    )
    
    # Training loop
    print(f'Starting training from epoch {start_epoch}')
    
    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, epoch, args, writer
        )
        
        # Validate
        if epoch % args.val_interval == 0:
            val_metrics = validate(model, val_loader, criterion, epoch, args)
            
            # Log validation metrics
            writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
            writer.add_scalar('Val/IoU', val_metrics['iou'], epoch)
            writer.add_scalar('Val/Precision', val_metrics['precision'], epoch)
            writer.add_scalar('Val/Recall', val_metrics['recall'], epoch)
            writer.add_scalar('Val/F1', val_metrics['f1'], epoch)
            
            print(f'Epoch {epoch} - Val Loss: {val_metrics["loss"]:.4f}, '
                  f'IoU: {val_metrics["iou"]:.4f}, '
                  f'F1: {val_metrics["f1"]:.4f}')
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, val_metrics, args)
        
        # Update learning rate
        scheduler.step()
        writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], epoch)
    
    writer.close()
    print('Training completed!')


if __name__ == '__main__':
    main()