"""
Simple Test Scripts for JIT and Quantized GDNet Models
Test your optimized models easily
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import os
import sys

# Add path for model imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# TEST 1: JIT OPTIMIZATION
# ============================================================================

def test_jit_optimization(checkpoint_path='./checkpoints/gdnet_best.pth', 
                          test_image='./data/test/image/700.jpg'):
    """
    Test JIT optimization on your GDNet model
    """
 
    print("TESTING JIT OPTIMIZATION")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Step 1: Load your trained GDNet model
    print("\n1. Loading original model...")
    from models.GDNet import GDNet
    
    model = GDNet(backbone='resnext101_32x8d', pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f" Model loaded (IoU: {checkpoint.get('metrics', {}).get('iou', 'N/A')})")
    
    # Step 2: Create JIT version
    print("\n2. Creating JIT optimized version...")
    example_input = torch.randn(1, 3, 384, 384).to(device)
    
    with torch.no_grad():
        jit_model = torch.jit.trace(model, example_input)
        jit_model = torch.jit.optimize_for_inference(jit_model)
    
    # Save JIT model
    torch.jit.save(jit_model, './checkpoints/gdnet_jit.pt')
    print(" JIT model saved to 'gdnet_jit.pt'")
    
    # Step 3: Compare file sizes
    original_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    jit_size = os.path.getsize('gdnet_jit.pt') / (1024 * 1024)
    print(f"\n3. File sizes:")
    print(f"   Original: {original_size:.2f} MB")
    print(f"   JIT:      {jit_size:.2f} MB")
    
    # Step 4: Speed comparison
    print("\n4. Speed comparison (100 runs)...")
    test_input = torch.randn(1, 3, 384, 384).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(test_input)
        _ = jit_model(test_input)
    
    # Time original model
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model(test_input)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    original_time = (time.time() - start) / 100 * 1000
    
    # Time JIT model
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = jit_model(test_input)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    jit_time = (time.time() - start) / 100 * 1000
    
    print(f"   Original: {original_time:.2f}ms ({1000/original_time:.1f} FPS)")
    print(f"   JIT:      {jit_time:.2f}ms ({1000/jit_time:.1f} FPS)")
    print(f"   Speedup:  {original_time/jit_time:.2f}×")
    
    # Step 5: Test on real image
    if os.path.exists(test_image):
        print(f"\n5. Testing on real image: {test_image}")
        
        # Load and preprocess image
        img = Image.open(test_image).convert('RGB')
        img_resized = img.resize((384, 384))
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        img_array = (img_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Original model inference
        start = time.time()
        with torch.no_grad():
            output_original = model(img_tensor)
        original_inference = (time.time() - start) * 1000
        
        # JIT model inference
        start = time.time()
        with torch.no_grad():
            output_jit = jit_model(img_tensor)
        jit_inference = (time.time() - start) * 1000
        
        print(f"   Original inference: {original_inference:.2f}ms")
        print(f"   JIT inference:      {jit_inference:.2f}ms")
        print(f"   Speedup:            {original_inference/jit_inference:.2f}×")
        
        # Check if outputs are identical
        diff = torch.abs(output_original - output_jit).max().item()
        print(f"   Max difference:     {diff:.6f} (should be ~0)")
        
        # Visualize
        prob_original = torch.sigmoid(output_original).cpu().numpy()[0, 0]
        prob_jit = torch.sigmoid(output_jit).cpu().numpy()[0, 0]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(prob_original, cmap='hot')
        axes[1].set_title(f'Original Model ({original_inference:.1f}ms)')
        axes[1].axis('off')
        
        axes[2].imshow(prob_jit, cmap='hot')
        axes[2].set_title(f'JIT Model ({jit_inference:.1f}ms)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('./results/optimization/jit_comparison.png')
        plt.show()
        print("    Visualization saved to 'jit_comparison.png'")
    
    print("\n JIT optimization test complete!")
    return jit_model


# ============================================================================
# TEST 2: QUANTIZATION (INT8)
# ============================================================================

def test_quantization(checkpoint_path='./checkpoints/gdnet_best.pth',
                     test_image='./data/test/image/700.jpg'):
    """
    Test INT8 Quantization on your GDNet model (CPU only)
    """
    device = torch.device('cpu')  # Quantization is for CPU
    
    # Step 1: Load model on CPU
    print("1. Loading original model on CPU...")
    from models.GDNet import GDNet
    
    model = GDNet(backbone='resnext101_32x8d', pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✓ Model loaded (IoU: {checkpoint.get('metrics', {}).get('iou', 'N/A')})")
    
    # Step 2: Apply quantization
    print("\n2. Applying INT8 quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Conv2d, nn.Linear},  # Quantize these layer types
        dtype=torch.qint8
    )
    
    # Save quantized model
    torch.save(quantized_model.state_dict(), './checkpoints/gdnet_quantized.pth')
    print(" Quantized model saved to 'gdnet_quantized.pth'")
    
    # Step 3: Compare model sizes
    def get_model_size_mb(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / 1024 / 1024
    
    original_size = get_model_size_mb(model)
    quantized_size = get_model_size_mb(quantized_model)
    
    print(f"\n3. Model sizes:")
    print(f"   Original:  {original_size:.2f} MB")
    print(f"   Quantized: {quantized_size:.2f} MB")
    print(f"   Reduction: {original_size/quantized_size:.2f}×")
    
    # Step 4: Speed comparison on CPU
    print("\n4. CPU Speed comparison (20 runs)...")
    test_input = torch.randn(1, 3, 384, 384)
    
    # Time original model
    start = time.time()
    for _ in range(20):
        with torch.no_grad():
            _ = model(test_input)
    original_time = (time.time() - start) / 20 * 1000
    
    # Time quantized model
    start = time.time()
    for _ in range(20):
        with torch.no_grad():
            _ = quantized_model(test_input)
    quantized_time = (time.time() - start) / 20 * 1000
    
    print(f"   Original:  {original_time:.2f}ms ({1000/original_time:.1f} FPS)")
    print(f"   Quantized: {quantized_time:.2f}ms ({1000/quantized_time:.1f} FPS)")
    print(f"   Speedup:   {original_time/quantized_time:.2f}×")
    
    # Step 5: Test on real image
    if os.path.exists(test_image):
        print(f"\n5. Testing on real image: {test_image}")
        
        # Load and preprocess
        img = Image.open(test_image).convert('RGB')
        img_resized = img.resize((384, 384))
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        img_array = (img_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1).unsqueeze(0)
        
        # Original inference
        start = time.time()
        with torch.no_grad():
            output_original = model(img_tensor)
        original_inference = (time.time() - start) * 1000
        
        # Quantized inference
        start = time.time()
        with torch.no_grad():
            output_quantized = quantized_model(img_tensor)
        quantized_inference = (time.time() - start) * 1000
        
        print(f"   Original inference:  {original_inference:.2f}ms")
        print(f"   Quantized inference: {quantized_inference:.2f}ms")
        print(f"   Speedup:             {original_inference/quantized_inference:.2f}×")
        
        # Calculate accuracy difference
        prob_original = torch.sigmoid(output_original).numpy()[0, 0]
        prob_quantized = torch.sigmoid(output_quantized).numpy()[0, 0]
        
        diff = np.abs(prob_original - prob_quantized).mean()
        print(f"   Mean difference:     {diff:.4f}")
        
        # Visualize
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # First row
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(prob_original, cmap='hot')
        axes[0, 1].set_title(f'Original Model ({original_inference:.1f}ms)')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(prob_quantized, cmap='hot')
        axes[0, 2].set_title(f'Quantized Model ({quantized_inference:.1f}ms)')
        axes[0, 2].axis('off')
        
        # Second row - binary masks
        mask_original = (prob_original > 0.5).astype(np.uint8) * 255
        mask_quantized = (prob_quantized > 0.5).astype(np.uint8) * 255
        
        axes[1, 0].axis('off')  # Empty
        
        axes[1, 1].imshow(mask_original, cmap='gray')
        axes[1, 1].set_title('Original Mask')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(mask_quantized, cmap='gray')
        axes[1, 2].set_title('Quantized Mask')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('./results/optimization/quantization_comparison.png')
        plt.show()
        print("    Visualization saved to 'quantization_comparison.png'")
    
    print("\n Quantization test complete!")
    return quantized_model


# ============================================================================
# LOAD AND USE SAVED MODELS
# ============================================================================

def load_and_use_jit_model(jit_path='gdnet_jit.pt', test_image='./data/test/image/700.jpg'):
    """
    Load and use a saved JIT model
    """
    print("Loading saved JIT model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load JIT model
    jit_model = torch.jit.load(jit_path, map_location=device)
    jit_model.eval()
    
    # Test on image
    img = Image.open(test_image).convert('RGB')
    img_resized = img.resize((384, 384))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_array = (img_array - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Inference
    start = time.time()
    with torch.no_grad():
        output = jit_model(img_tensor)
    inference_time = (time.time() - start) * 1000
    
    prob = torch.sigmoid(output).cpu().numpy()[0, 0]
    mask = (prob > 0.5).astype(np.uint8) * 255
    
    print(f"Inference time: {inference_time:.2f}ms")
    print(f"Glass detected: {np.sum(mask > 0) / mask.size * 100:.1f}%")
    
    return prob, mask


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test JIT and Quantization optimizations')
    parser.add_argument('--mode', type=str, default='both',
                       choices=['jit', 'quantized', 'both'],
                       help='Which optimization to test')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/gdnet_best.pth',
                       help='Path to GDNet checkpoint')
    parser.add_argument('--image', type=str, default='./data/test/image/700.jpg',
                       help='Test image path')
    parser.add_argument('--load-jit', type=str, default=None,
                       help='Load existing JIT model')
    
    args = parser.parse_args()
    
    if args.load_jit:
        # Load and use existing JIT model
        prob, mask = load_and_use_jit_model(args.load_jit, args.image)
        
    elif args.mode == 'jit' or args.mode == 'both':
        # Test JIT optimization
        jit_model = test_jit_optimization(args.checkpoint, args.image)
        
    if args.mode == 'quantized' or args.mode == 'both':
        # Test quantization
        print("\n" + "="*60 + "\n")
        quantized_model = test_quantization(args.checkpoint, args.image)
        
    if args.mode == 'both':
        print("\n" + "="*60)
        print("SUMMARY")
        print("\nFiles created:")
        print("   - gdnet_jit.pt (JIT optimized model)")
        print("   - gdnet_quantized.pth (Quantized model)")
        print("   - jit_comparison.png")
        print("   - quantization_comparison.png")


if __name__ == '__main__':
    main()