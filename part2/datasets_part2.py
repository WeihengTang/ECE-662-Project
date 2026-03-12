"""
Data loaders for Part 2: MNIST → SVHN domain adaptation.

Key handling:
  - MNIST  : 1-channel 28×28 → resized to 32×32 (to match Part 1 AlexNet).
  - SVHN   : 3-channel 32×32 → averaged to 1-channel (per project spec).

The channel-averaging transform converts an RGB PIL image to grayscale by
taking the arithmetic mean of the three channels:
    gray = (R + G + B) / 3

This is distinct from the luminance-weighted conversion used by
torchvision.transforms.Grayscale() (which applies 0.299R + 0.587G + 0.114B).
We use the true mean to match the project description exactly.
"""
import os
import sys
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR


# ── Normalisation constants ──────────────────────────────────────────────────
# MNIST standard stats (1-channel)
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)

# SVHN grayscale stats (approximate, computed by averaging channels).
# True-average-grayscale SVHN ≈ mean 0.4524, std 0.2067.
SVHN_MEAN, SVHN_STD = (0.4524,), (0.2067,)


def _rgb_avg_to_gray(pil_img: Image.Image) -> Image.Image:
    """Average the three channels of an RGB PIL image to produce 1-channel."""
    arr = np.array(pil_img, dtype=np.float32)   # (H, W, 3)
    gray = arr.mean(axis=2).astype(np.uint8)     # (H, W)
    return Image.fromarray(gray, mode='L')


def get_mnist_loaders(batch_size: int = 128, data_root: str = './data',
                      num_workers: int = 4):
    """Return (train_loader, test_loader) for MNIST (1-ch, 32×32)."""
    tfm = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    train_ds = datasets.MNIST(data_root, train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST(data_root, train=False, download=True, transform=tfm)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True),
        DataLoader(test_ds,  batch_size=256, shuffle=False,
                   num_workers=num_workers, pin_memory=True),
    )


def get_svhn_loaders(batch_size: int = 128, data_root: str = './data',
                     num_workers: int = 4):
    """
    Return (train_loader, test_loader) for SVHN converted to 1-channel 32×32.

    The transform pipeline:
        1. PIL RGB image (32×32×3) from torchvision SVHN
        2. Channel-average to grayscale (32×32×1)
        3. ToTensor → (1, 32, 32) in [0, 1]
        4. Normalize with SVHN_MEAN / SVHN_STD
    """
    tfm = transforms.Compose([
        transforms.Lambda(_rgb_avg_to_gray),   # RGB mean → 1-channel PIL
        transforms.ToTensor(),                  # (1, 32, 32) in [0,1]
        transforms.Normalize(SVHN_MEAN, SVHN_STD),
    ])
    train_ds = datasets.SVHN(data_root, split='train', download=True, transform=tfm)
    test_ds  = datasets.SVHN(data_root, split='test',  download=True, transform=tfm)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True),
        DataLoader(test_ds,  batch_size=256, shuffle=False,
                   num_workers=num_workers, pin_memory=True),
    )


def get_svhn_test_loader(batch_size: int = 256, data_root: str = './data',
                         num_workers: int = 4):
    """Return test loader only (for evaluation of MNIST-trained models on SVHN)."""
    return get_svhn_loaders(batch_size=batch_size, data_root=data_root,
                            num_workers=num_workers)[1]
