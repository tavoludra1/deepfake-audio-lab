"""
Deep Learning Architectures for Sonic Signature Verification.
Standard: Microsoft Research / NeurIPS high-level implementation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SonicIdentityNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=2):
        super(SonicIdentityNet, self).__init__()
        # Convolutional Backbone for Spectral Features
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Global Average Pooling to handle variable audio lengths
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dense layers with Dropout for regularization
        self.fc1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input Check (NASA Standard)
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got {x.dim()}D")

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
