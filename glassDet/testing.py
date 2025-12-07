"""
Glass Detection with Bounding Box Visualization
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GlassDetectorWithBBox:
    """Glass detection with bounding box generation"""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(checkpoint_path)
        print(f"Model loaded on {self.device}")
    
    def _load_model(self, checkpoint_path):
        """Load the trained GDNet model"""
        # Import here to avoid circular import issues
        from models.GDNet import GDNet
        
        model = GDNet(backbone='resnext101_32x8d', pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'metrics' in checkpoint:
                metrics = checkpoint['metrics']
                print(f"Model Performance - IoU: {metrics.get('iou', 0):.4f}, F1: {metrics.get('f1', 0):.4f}")
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def preprocess_image(self, image_path):
        """Load and preprocess image"""
        # Load image
        img = Image.open(image_path).convert('RGB')
        original_img = np.array(img)
        original_size = img.size
        
        # Resize for model
        img_resized = img.resize((384, 384))
        
        # Convert to tensor
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        img_tensor = torch.from_numpy(img_array).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor.to(self.device), original_img, original_size
    
    def detect_glass(self, image_path, threshold=0.5):
        """Detect glass regions and generate bounding boxes"""
        # Preprocess image
        img_tensor, original_img, original_size = self.preprocess_image(image_path)
        
        # Predict glass regions
        with torch.no_grad():
            output = self.model(img_tensor)
            prob = torch.sigmoid(output).cpu().numpy()[0, 0]
        
        # Resize probability map to original size
        prob_resized = cv2.resize(prob, original_size, interpolation=cv2.INTER_LINEAR)
        
        # Create binary mask
        binary_mask = (prob_resized > threshold).astype(np.uint8) * 255
        
        # Get bounding boxes
        bboxes = self.get_bounding_boxes(binary_mask)
        
        # Filter small boxes
        min_area = 0.001 * binary_mask.shape[0] * binary_mask.shape[1]  # 0.1% of image
        bboxes = [box for box in bboxes if box['area'] > min_area]
        
        return {
            'image': original_img,
            'probability_map': prob_resized,
            'binary_mask': binary_mask,
            'bounding_boxes': bboxes,
            'glass_ratio': np.sum(binary_mask > 0) / binary_mask.size
        }
    
    def get_bounding_boxes(self, binary_mask):
        """Extract bounding boxes from binary mask"""
        # Find contours
        contours, hierarchy = cv2.findContours(
            binary_mask, 
            cv2.RETR_EXTERNAL,  # Only external contours
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        bboxes = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate area and confidence
            area = cv2.contourArea(contour)
            bbox_area = w * h
            fill_ratio = area / bbox_area if bbox_area > 0 else 0
            
            bboxes.append({
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'area': area,
                'bbox_area': bbox_area,
                'fill_ratio': fill_ratio,
                'confidence': fill_ratio,  # Use fill ratio as confidence
                'contour': contour
            })
        
        # Sort by area (largest first)
        bboxes.sort(key=lambda x: x['area'], reverse=True)
        
        return bboxes
    
    def visualize_results(self, results, save_path=None, show=True):
        """Visualize detection results with bounding boxes"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        image = results['image']
        prob_map = results['probability_map']
        mask = results['binary_mask']
        bboxes = results['bounding_boxes']
        
        # 1. Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 2. Probability map
        im = axes[0, 1].imshow(prob_map, cmap='hot')
        axes[0, 1].set_title('Glass Probability Map')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
        
        # 3. Binary mask
        axes[0, 2].imshow(mask, cmap='gray')
        axes[0, 2].set_title(f'Binary Mask ({results["glass_ratio"]:.1%} glass)')
        axes[0, 2].axis('off')
        
        # 4. Bounding boxes on original
        axes[1, 0].imshow(image)
        for i, bbox in enumerate(bboxes):
            rect = patches.Rectangle(
                (bbox['x'], bbox['y']), bbox['width'], bbox['height'],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            axes[1, 0].add_patch(rect)
            # Add label
            axes[1, 0].text(bbox['x'], bbox['y']-5, f'Glass {i+1}', 
                          color='red', fontsize=10, fontweight='bold',
                          bbox=dict(facecolor='white', alpha=0.7))
        axes[1, 0].set_title(f'Detected Glass Regions ({len(bboxes)} objects)')
        axes[1, 0].axis('off')
        
        # 5. Overlay with bounding boxes
        overlay = image.copy()
        mask_colored = np.zeros_like(image)
        mask_colored[:, :, 0] = mask  # Red channel for glass
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        axes[1, 1].imshow(overlay)
        for bbox in bboxes:
            rect = patches.Rectangle(
                (bbox['x'], bbox['y']), bbox['width'], bbox['height'],
                linewidth=2, edgecolor='yellow', facecolor='none'
            )
            axes[1, 1].add_patch(rect)
        axes[1, 1].set_title('Overlay with Bounding Boxes')
        axes[1, 1].axis('off')
        
        # 6. Detection statistics
        axes[1, 2].axis('off')
        stats_text = f"Detection Statistics:\n\n"
        stats_text += f"Total glass objects: {len(bboxes)}\n"
        stats_text += f"Glass coverage: {results['glass_ratio']:.1%}\n\n"
        
        for i, bbox in enumerate(bboxes[:5]):  # Show top 5
            stats_text += f"Object {i+1}:\n"
            stats_text += f"  Size: {bbox['width']}×{bbox['height']} px\n"
            stats_text += f"  Area: {bbox['area']:.0f} px²\n"
            stats_text += f"  Confidence: {bbox['confidence']:.2f}\n\n"
        
        axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top',
                       fontfamily='monospace')
        axes[1, 2].set_title('Detection Details')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Results saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def draw_boxes_on_image(self, image, bboxes, color=(0, 255, 0), thickness=2):
        """Draw bounding boxes directly on image using OpenCV"""
        img_with_boxes = image.copy()
        
        for i, bbox in enumerate(bboxes):
            # Draw rectangle
            cv2.rectangle(
                img_with_boxes,
                (bbox['x'], bbox['y']),
                (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                color,
                thickness
            )
            
            # Add label with background
            label = f"Glass {i+1}: {bbox['confidence']:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw label background
            cv2.rectangle(
                img_with_boxes,
                (bbox['x'], bbox['y'] - label_size[1] - 4),
                (bbox['x'] + label_size[0], bbox['y']),
                color, -1
            )
            
            # Draw label text
            cv2.putText(
                img_with_boxes,
                label,
                (bbox['x'], bbox['y'] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return img_with_boxes
    
    def process_video(self, video_path, output_path=None):
        """Process video and add bounding boxes to each frame"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Save frame temporarily
            temp_path = 'temp_frame.jpg'
            cv2.imwrite(temp_path, frame)
            
            # Detect glass
            results = self.detect_glass(temp_path)
            
            # Draw boxes on frame
            frame_with_boxes = self.draw_boxes_on_image(
                frame, results['bounding_boxes'], color=(0, 255, 0)
            )
            
            # Show or save
            if output_path:
                out.write(frame_with_boxes)
            
            # Display (optional)
            cv2.imshow('Glass Detection', frame_with_boxes)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Clean up temp file
            os.remove(temp_path)
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"Processed {frame_count} frames")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Glass Detection with Bounding Boxes')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Detection threshold (0-1)')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save visualization')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display results')
    
    args = parser.parse_args()
    
    # Create detector
    detector = GlassDetectorWithBBox(args.checkpoint)
    
    # Detect glass
    print(f"Processing: {args.image}")
    results = detector.detect_glass(args.image, threshold=args.threshold)
    
    # Print detection results
    print(f"\nDetection Results:")
    print(f"  Glass regions found: {len(results['bounding_boxes'])}")
    print(f"  Glass coverage: {results['glass_ratio']:.1%}")
    
    for i, bbox in enumerate(results['bounding_boxes']):
        print(f"\nGlass Object {i+1}:")
        print(f"  Position: ({bbox['x']}, {bbox['y']})")
        print(f"  Size: {bbox['width']}×{bbox['height']} pixels")
        print(f"  Area: {bbox['area']:.0f} pixels²")
        print(f"  Confidence: {bbox['confidence']:.2f}")
    
    # Visualize results
    detector.visualize_results(
        results, 
        save_path=args.save,
        show=not args.no_show
    )


if __name__ == '__main__':
    main()