"""
AlexNet-DCF for MNIST.

Every Conv2d in AlexNetMNIST is replaced by Conv_DCF with learnable atoms
(bases_grad=True) and learnable coefficients.

This corresponds to the "DCF from scratch" model in Tasks 1 & 2, where both
ψ_k (atoms) and a_k (coefficients) are learned jointly during training.

Usage:
    model = AlexNetDCF(num_bases=4, bases_grad=True)   # learnable atoms
    model = AlexNetDCF(num_bases=4, bases_grad=False)  # FB-initialised, fixed atoms
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from dcf_layer import Conv_DCF


class AlexNetDCF(nn.Module):
    """
    AlexNet for MNIST where every conv layer is replaced by Conv_DCF.

    Args:
        num_bases  (int) : K — number of filter atoms per conv layer.
        bases_grad (bool): If True, atoms are learnable (DCF from scratch).
                           If False, atoms are fixed (FB init, only coefficients learned).
        initializer (str): 'FB' or 'random'.
        num_classes (int): Number of output classes. Default 10.
    """

    def __init__(self, num_bases: int = 8, bases_grad: bool = True,
                 initializer: str = 'FB', num_classes: int = 10):
        super().__init__()
        self.num_bases  = num_bases
        self.bases_grad = bases_grad

        def make_dcf(in_ch, out_ch):
            return Conv_DCF(in_ch, out_ch, kernel_size=3, padding=1,
                            num_bases=num_bases, bias=False,
                            bases_grad=bases_grad, initializer=initializer,
                            mode='mode1')

        self.features = nn.Sequential(
            # Block 1
            make_dcf(1, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            make_dcf(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            make_dcf(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Block 4
            make_dcf(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Block 5
            make_dcf(128, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
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
        return self.classifier(x)

    def count_conv_params(self) -> int:
        """Count trainable parameters in DCF conv layers only."""
        total = 0
        for m in self.modules():
            if isinstance(m, Conv_DCF):
                total += m.num_trainable_params()
        return total
