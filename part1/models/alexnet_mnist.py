"""
AlexNet adapted for MNIST (1-channel, 32×32 input after resize).

Architecture mirrors the 5-conv-layer spirit of the original AlexNet but is
scaled down to handle the small 32×32 input:

    Conv1 (1→32, 3×3, pad=1) + BN + ReLU → Pool (2×2)   [32×16×16]
    Conv2 (32→64, 3×3, pad=1) + BN + ReLU → Pool (2×2)  [64×8×8]
    Conv3 (64→128, 3×3, pad=1) + BN + ReLU              [128×8×8]
    Conv4 (128→128, 3×3, pad=1) + BN + ReLU             [128×8×8]
    Conv5 (128→64, 3×3, pad=1) + BN + ReLU → Pool (2×2) [64×4×4]
    FC(1024→512) + FC(512→256) + FC(256→10)
"""
import torch
import torch.nn as nn


class AlexNetMNIST(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                          # → 32×16×16
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                          # → 64×8×8
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Block 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Block 5
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                          # → 64×4×4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    # ── Convenience: layer references used by run_pca.py ────────────────────
    @property
    def conv_layers(self):
        """Return a list of (name, nn.Conv2d) tuples for each conv layer."""
        return [
            (f'features.{i}', m)
            for i, m in enumerate(self.features)
            if isinstance(m, nn.Conv2d)
        ]
