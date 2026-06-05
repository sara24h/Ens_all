"""
models.py
=========
ResNet-50 wrapper with intermediate feature & attention-map extraction.

Used by both teacher and student networks in all three KD methods:
  - Response-based (Logits)  → needs logits only
  - Feature-based (AT)       → needs attention maps from layer1–4
  - Relation-based (RKD)     → needs penultimate features (avgpool output)
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet50WithFeatures(nn.Module):
    """
    ResNet-50 that can optionally return:
      - penultimate features  [B, 2048]   (for RKD)
      - attention maps from layers 1-4     (for AT)

    Forward signature
    -----------------
    >>> logits = model(x)
    >>> logits, features = model(x, return_features=True)
    >>> logits, attn_maps = model(x, return_attention=True)
    >>> logits, features, attn_maps = model(x, return_features=True, return_attention=True)
    """

    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()

        backbone = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )

        # --- copy sub-modules ---
        self.conv1   = backbone.conv1
        self.bn1     = backbone.bn1
        self.relu    = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1  = backbone.layer1     # 256 ch
        self.layer2  = backbone.layer2     # 512 ch
        self.layer3  = backbone.layer3     # 1024 ch
        self.layer4  = backbone.layer4     # 2048 ch
        self.avgpool = backbone.avgpool
        self.fc      = nn.Linear(backbone.fc.in_features, num_classes)

        self.feature_channels = [256, 512, 1024, 2048]

    def forward(self, x,
                return_features: bool = False,
                return_attention: bool = False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        attention_maps = []
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
            if return_attention:
                attention_maps.append(x)

        features = self.avgpool(x)
        features = torch.flatten(features, 1)   # [B, 2048]
        logits   = self.fc(features)             # [B, num_classes]

        # --- build return tuple ---
        if not return_features and not return_attention:
            return logits

        parts = [logits]
        if return_features:
            parts.append(features)
        if return_attention:
            parts.append(attention_maps)
        return tuple(parts)
