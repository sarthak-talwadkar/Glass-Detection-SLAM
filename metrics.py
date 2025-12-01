import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def calculate_metrics(predictions, ground_truth):
    """
    Calculate evaluation metrics for glass detection
    
    Args:
        predictions: Binary predictions (B, 1, H, W)
        ground_truth: Ground truth masks (B, 1, H, W)
    
    Returns:
        Dictionary of metrics
    """
    # Flatten arrays
    pred_flat = predictions.flatten()
    gt_flat = ground_truth.flatten()
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_flat, pred_flat, average='binary'
    )
    
    # Calculate IoU
    intersection = np.logical_and(pred_flat, gt_flat).sum()
    union = np.logical_or(pred_flat, gt_flat).sum()
    iou = intersection / (union + 1e-6)
    
    # Calculate accuracy
    accuracy = (pred_flat == gt_flat).mean()
    
    # Calculate MAE (Mean Absolute Error)
    mae = np.abs(pred_flat - gt_flat).mean()
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
        'accuracy': accuracy,
        'mae': mae
    }


def calculate_boundary_metrics(predictions, ground_truth, tolerance=3):
    """
    Calculate boundary-specific metrics
    
    Args:
        predictions: Binary predictions
        ground_truth: Ground truth masks
        tolerance: Pixel tolerance for boundary matching
    
    Returns:
        Boundary F-measure
    """
    from scipy.ndimage import distance_transform_edt
    
    # Get boundaries
    pred_boundary = get_boundary_mask(predictions)
    gt_boundary = get_boundary_mask(ground_truth)
    
    # Calculate distance transforms
    pred_dist = distance_transform_edt(~pred_boundary)
    gt_dist = distance_transform_edt(~gt_boundary)
    
    # Calculate precision and recall with tolerance
    pred_match = pred_dist[gt_boundary] <= tolerance
    gt_match = gt_dist[pred_boundary] <= tolerance
    
    if pred_boundary.sum() == 0:
        precision = 0
    else:
        precision = gt_match.sum() / pred_boundary.sum()
    
    if gt_boundary.sum() == 0:
        recall = 0
    else:
        recall = pred_match.sum() / gt_boundary.sum()
    
    # Calculate F-measure
    if precision + recall == 0:
        f_measure = 0
    else:
        f_measure = 2 * precision * recall / (precision + recall)
    
    return f_measure


def get_boundary_mask(mask, kernel_size=5):
    """Extract boundary from binary mask"""
    import cv2
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    boundary = dilated - eroded
    
    return boundary.astype(bool)