"""
GDNet: Glass Detection Network
Paper: Don't Hit Me! Glass Detection in Real-World Scenes (CVPR 2020)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class LCFI(nn.Module):
    """Large-field Contextual Feature Integration Module"""
    
    def __init__(self, in_channels, out_channels):
        super(LCFI, self).__init__()
        
        # Multiple parallel dilated convolutions for different receptive fields
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=5, dilation=5),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=7, dilation=7),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Feature aggregation
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Multi-scale feature extraction
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat4 = self.branch4(x)
        
        # Concatenate all branches
        combined = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        
        # Feature fusion
        out = self.fusion(combined)
        
        return out
        
class DecoderBlock(nn.Module):
    """Decoder block with skip connections"""
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, skip=None):
        # Upsample
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        
        # Concatenate with skip connection if provided
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x


class GDNet(nn.Module):
    """Glass Detection Network"""
    
    def __init__(self, backbone='resnext101_32x8d', pretrained=True):
        super(GDNet, self).__init__()
        
        # Load backbone
        if backbone == 'resnext101_32x8d':
            self.backbone = models.resnext101_32x8d(pretrained=pretrained)
        else:
            self.backbone = models.resnet101(pretrained=pretrained)
        
        # Encoder layers from backbone
        self.encoder1 = nn.Sequential(*list(self.backbone.children())[:3])  # 64 channels
        self.encoder2 = nn.Sequential(*list(self.backbone.children())[3:5])  # 256 channels
        self.encoder3 = list(self.backbone.children())[5]  # 512 channels
        self.encoder4 = list(self.backbone.children())[6]  # 1024 channels
        self.encoder5 = list(self.backbone.children())[7]  # 2048 channels
        
        # LCFI modules for multi-scale feature enhancement
        self.lcfi5 = LCFI(2048, 512)
        self.lcfi4 = LCFI(1024, 256)
        self.lcfi3 = LCFI(512, 128)
        self.lcfi2 = LCFI(256, 64)
        
        # Decoder blocks
        self.decoder5 = DecoderBlock(512, 256, 256)  # 512 -> 256
        self.decoder4 = DecoderBlock(256, 128, 128)  # 256 -> 128
        self.decoder3 = DecoderBlock(128, 64, 64)    # 128 -> 64
        self.decoder2 = DecoderBlock(64, 64, 32)     # 64 -> 32
        self.decoder1 = DecoderBlock(32, 0, 16)      # 32 -> 16
        
        # Final prediction head
        self.final = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Store input size for final upsampling
        input_size = x.shape[2:]
        
        # Encoder
        e1 = self.encoder1(x)    # 64, H/2, W/2
        e2 = self.encoder2(e1)   # 256, H/4, W/4
        e3 = self.encoder3(e2)   # 512, H/8, W/8
        e4 = self.encoder4(e3)   # 1024, H/16, W/16
        e5 = self.encoder5(e4)   # 2048, H/32, W/32
        
        # Apply LCFI modules for contextual feature enhancement
        lcfi5_out = self.lcfi5(e5)  # 512
        lcfi4_out = self.lcfi4(e4)  # 256
        lcfi3_out = self.lcfi3(e3)  # 128
        lcfi2_out = self.lcfi2(e2)  # 64
        
        # Decoder with skip connections
        d5 = self.decoder5(lcfi5_out, lcfi4_out)  # 256, H/16, W/16
        d4 = self.decoder4(d5, lcfi3_out)         # 128, H/8, W/8
        d3 = self.decoder3(d4, lcfi2_out)         # 64, H/4, W/4
        d2 = self.decoder2(d3, e1)                # 32, H/2, W/2
        d1 = self.decoder1(d2)                    # 16, H, W
        
        # Final prediction
        out = self.final(d1)
        
        # Upsample to input size
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        
        return out
    
    def get_intermediate_features(self, x):
        """Get intermediate features for visualization"""
        features = {}
        
        e1 = self.encoder1(x)
        features['encoder1'] = e1
        
        e2 = self.encoder2(e1)
        features['encoder2'] = e2
        
        e3 = self.encoder3(e2)
        features['encoder3'] = e3
        
        e4 = self.encoder4(e3)
        features['encoder4'] = e4
        
        e5 = self.encoder5(e4)
        features['encoder5'] = e5
        
        features['lcfi5'] = self.lcfi5(e5)
        features['lcfi4'] = self.lcfi4(e4)
        features['lcfi3'] = self.lcfi3(e3)
        features['lcfi2'] = self.lcfi2(e2)
        
        return features


# Test the model
if __name__ == "__main__":
    model = GDNet(backbone='resnext101_32x8d', pretrained=False)
    model.eval()
    
    # Test with random input
    x = torch.randn(2, 3, 384, 384)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")