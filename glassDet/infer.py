"""
Inference script for GDNet
"""

import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from models.GDNet import GDNet


def parse_args():
    parser = argparse.ArgumentParser(description='GDNet Inference')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Input image or directory path')
    parser.add_argument('--output', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--img-size', type=int, default=384,
                        help='Input image size')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary mask')
    parser.add_argument('--save-overlay', action='store_true',
                        help='Save overlay visualization')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU id to use')
    parser.add_argument('--crf', action='store_true',
                        help='Apply CRF post-processing')
    
    return parser.parse_args()


class GDNetInference:
    """Glass Detection Network Inference"""
    
    def __init__(self, checkpoint_path, img_size=384, device='cuda'):
        self.img_size = img_size
        self.device = device
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, checkpoint_path):
        """Load model from checkpoint"""
        # Create model
        model = GDNet(backbone='resnext101_32x8d', pretrained=False)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        print(f'Model loaded from {checkpoint_path}')
        
        return model
    
    def preprocess(self, image_path):
        """Preprocess input image"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        return image_tensor, original_size
    
    def postprocess(self, output, original_size, threshold=0.5):
        """Postprocess model output"""
        # Apply sigmoid
        output = torch.sigmoid(output)
        
        # Convert to numpy
        output = output.squeeze().cpu().numpy()
        
        # Resize to original size
        output = cv2.resize(output, original_size, interpolation=cv2.INTER_LINEAR)
        
        # Apply threshold
        binary_mask = (output > threshold).astype(np.uint8) * 255
        
        return output, binary_mask
    
    def apply_crf(self, image, prob_map):
        """Apply CRF post-processing (requires dss_crf)"""
        try:
            from dss_crf import crf_refine
            
            # Convert probability map to uint8
            prob_uint8 = (prob_map * 255).astype(np.uint8)
            
            # Apply CRF
            refined = crf_refine(image, prob_uint8)
            
            return refined
        except ImportError:
            print("Warning: dss_crf not installed. Skipping CRF refinement.")
            return (prob_map * 255).astype(np.uint8)
    
    def create_overlay(self, image, mask, alpha=0.5):
        """Create overlay visualization"""
        # Load original image
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[:, :, 0] = mask  # Red channel for glass regions
        
        # Create overlay
        overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
        
        return overlay
    
    def predict(self, image_path, threshold=0.5, use_crf=False):
        """Perform inference on single image"""
        # Preprocess
        image_tensor, original_size = self.preprocess(image_path)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(image_tensor)
        
        # Postprocess
        prob_map, binary_mask = self.postprocess(output, original_size, threshold)
        
        # Apply CRF if requested
        if use_crf:
            image = cv2.imread(image_path)
            binary_mask = self.apply_crf(image, prob_map)
        
        return prob_map, binary_mask
    
    def predict_batch(self, image_paths, batch_size=8):
        """Perform inference on multiple images"""
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            original_sizes = []
            
            # Prepare batch
            for path in batch_paths:
                tensor, size = self.preprocess(path)
                batch_tensors.append(tensor)
                original_sizes.append(size)
            
            # Stack tensors
            batch = torch.cat(batch_tensors, dim=0)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(batch)
            
            # Process each output
            for j, output in enumerate(outputs):
                prob_map, binary_mask = self.postprocess(
                    output.unsqueeze(0), 
                    original_sizes[j]
                )
                results.append((prob_map, binary_mask))
        
        return results


def process_single_image(args, inferencer):
    """Process single image"""
    print(f'Processing: {args.input}')
    
    # Predict
    prob_map, binary_mask = inferencer.predict(
        args.input, 
        threshold=args.threshold,
        use_crf=args.crf
    )
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    
    # Save binary mask
    mask_name = os.path.basename(args.input).replace('.jpg', '_mask.png')
    mask_path = os.path.join(args.output, mask_name)
    cv2.imwrite(mask_path, binary_mask)
    print(f'Mask saved: {mask_path}')
    
    # Save probability map
    prob_name = os.path.basename(args.input).replace('.jpg', '_prob.png')
    prob_path = os.path.join(args.output, prob_name)
    prob_uint8 = (prob_map * 255).astype(np.uint8)
    cv2.imwrite(prob_path, prob_uint8)
    print(f'Probability map saved: {prob_path}')
    
    # Save overlay if requested
    if args.save_overlay:
        overlay = inferencer.create_overlay(args.input, binary_mask)
        overlay_name = os.path.basename(args.input).replace('.jpg', '_overlay.png')
        overlay_path = os.path.join(args.output, overlay_name)
        plt.imsave(overlay_path, overlay)
        print(f'Overlay saved: {overlay_path}')
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    original = Image.open(args.input)
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Probability map
    axes[1].imshow(prob_map, cmap='hot')
    axes[1].set_title('Probability Map')
    axes[1].axis('off')
    
    # Binary mask
    axes[2].imshow(binary_mask, cmap='gray')
    axes[2].set_title('Binary Mask')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'results.png'))
    plt.show()


def process_directory(args, inferencer):
    """Process all images in directory"""
    # Get all image files
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in extensions:
        image_files.extend([
            os.path.join(args.input, f) 
            for f in os.listdir(args.input) 
            if f.lower().endswith(ext)
        ])
    
    print(f'Found {len(image_files)} images')
    
    # Process each image
    os.makedirs(args.output, exist_ok=True)
    
    for image_path in tqdm(image_files, desc='Processing images'):
        # Predict
        prob_map, binary_mask = inferencer.predict(
            image_path,
            threshold=args.threshold,
            use_crf=args.crf
        )
        
        # Save results
        base_name = os.path.basename(image_path).split('.')[0]
        
        # Save binary mask
        mask_path = os.path.join(args.output, f'{base_name}_mask.png')
        cv2.imwrite(mask_path, binary_mask)
        
        # Save probability map
        prob_path = os.path.join(args.output, f'{base_name}_prob.png')
        prob_uint8 = (prob_map * 255).astype(np.uint8)
        cv2.imwrite(prob_path, prob_uint8)
        
        # Save overlay if requested
        if args.save_overlay:
            overlay = inferencer.create_overlay(image_path, binary_mask)
            overlay_path = os.path.join(args.output, f'{base_name}_overlay.png')
            plt.imsave(overlay_path, overlay)
    
    print(f'Results saved to {args.output}')


def main():
    args = parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create inferencer
    inferencer = GDNetInference(
        checkpoint_path=args.checkpoint,
        img_size=args.img_size,
        device=device
    )
    
    # Check if input is file or directory
    if os.path.isfile(args.input):
        process_single_image(args, inferencer)
    elif os.path.isdir(args.input):
        process_directory(args, inferencer)
    else:
        raise ValueError(f'Invalid input path: {args.input}')


if __name__ == '__main__':
    main()