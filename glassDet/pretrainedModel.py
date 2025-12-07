# download_backbone.py
import torch
import torchvision.models as models

# Download pretrained ResNeXt
model = models.resnext101_32x8d(pretrained=True)

# Save the state dict
torch.save(model.state_dict(), './checkpoints/resnext101_32x8d_pretrained.pth')
print("Backbone saved to ./checkpoints/resnext101_32x8d_pretrained.pth")