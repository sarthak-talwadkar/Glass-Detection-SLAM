# test_installation.py
import sys
import torch
import torchvision
import cv2
import numpy as np

def test_installation():
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
    print("OpenCV version:", cv2.__version__)
    print("NumPy version:", np.__version__)
    
    # Test basic tensor operation
    try:
        x = torch.randn(2, 3, 224, 224)
        if torch.cuda.is_available():
            x = x.cuda()
            print(" CUDA tensor operations working")
        print(" Basic tensor operations working")
        return True
    except Exception as e:
        print(f" Error: {e}")
        return False

if __name__ == "__main__":
    test_installation()