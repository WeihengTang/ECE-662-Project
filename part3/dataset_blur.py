"""
Synthetic Spatially-Varying Blur Tasks on MNIST.

Generates three sequential continual-learning tasks, each degrading clean
MNIST digits with a distinct blur type:

  Task 0 — Gaussian blur  (isotropic PSF, σ = 1.5)
  Task 1 — Motion blur    (linear 45° PSF, length = 5 px)
  Task 2 — Defocus blur   (disk / pillbox PSF, radius = 2.5 px)

Each task supplies (blurred_image, clean_image) pairs for training a
deblurring autoencoder.  All images are in [0, 1], single-channel, 28×28.

The blur kernels are applied via 2-D convolution (F.conv2d) with reflect
padding so that the output size matches the input.

Usage:
    from dataset_blur import get_blur_task_loaders
    train_ld, test_ld = get_blur_task_loaders(task_id=0, batch_size=128)
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


# ═══════════════════════════════════════════════════════════════════════════════
# Blur kernel constructors
# ═══════════════════════════════════════════════════════════════════════════════

def _gaussian_kernel(size: int = 7, sigma: float = 1.5) -> torch.Tensor:
    """Isotropic 2-D Gaussian kernel, normalised to sum 1."""
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g.outer(g)
    return kernel / kernel.sum()


def _motion_kernel(size: int = 7, angle_deg: float = 45.0) -> torch.Tensor:
    """Linear motion-blur kernel at a given angle, normalised to sum 1."""
    kernel = torch.zeros(size, size)
    centre = (size - 1) / 2.0
    angle_rad = math.radians(angle_deg)
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    length = (size - 1) / 2.0
    # Bresenham-style: draw a 1-pixel-wide line through the centre
    for t_i in range(200):
        t = -length + 2 * length * t_i / 199.0
        x = centre + t * dx
        y = centre + t * dy
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < size and 0 <= yi < size:
            kernel[yi, xi] = 1.0
    if kernel.sum() == 0:
        kernel[size // 2, size // 2] = 1.0
    return kernel / kernel.sum()


def _disk_kernel(size: int = 7, radius: float = 2.5) -> torch.Tensor:
    """Pillbox / disk (defocus) kernel, normalised to sum 1."""
    centre = (size - 1) / 2.0
    y, x = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing='ij',
    )
    dist = torch.sqrt((x - centre) ** 2 + (y - centre) ** 2)
    kernel = (dist <= radius).float()
    if kernel.sum() == 0:
        kernel[size // 2, size // 2] = 1.0
    return kernel / kernel.sum()


# Canonical task definitions: (name, kernel_fn, kwargs, kernel_size)
TASK_DEFS = [
    ('gaussian',  _gaussian_kernel, dict(size=7, sigma=1.5),            7),
    ('motion',    _motion_kernel,   dict(size=7, angle_deg=45.0),       7),
    ('defocus',   _disk_kernel,     dict(size=7, radius=2.5),           7),
]

NUM_TASKS = len(TASK_DEFS)


def get_blur_kernel(task_id: int) -> torch.Tensor:
    """Return the blur kernel for a given task as a (1,1,kH,kW) tensor."""
    name, fn, kwargs, _ = TASK_DEFS[task_id]
    kernel = fn(**kwargs)                      # (kH, kW)
    return kernel.unsqueeze(0).unsqueeze(0)    # (1, 1, kH, kW)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class BlurredMNIST(Dataset):
    """
    Wraps torchvision MNIST and returns (blurred, clean) pairs.

    The blur is applied on-the-fly (deterministic given the kernel).
    """

    def __init__(self, task_id: int, train: bool = True,
                 data_root: str = './data'):
        super().__init__()
        tfm = transforms.Compose([transforms.ToTensor()])
        self.mnist = datasets.MNIST(data_root, train=train,
                                    download=True, transform=tfm)
        self.kernel = get_blur_kernel(task_id)         # (1,1,kH,kW)
        self.pad = (self.kernel.shape[-1] - 1) // 2

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        clean, _label = self.mnist[idx]   # clean: (1, 28, 28) in [0,1]
        # Apply blur: reflect-pad then conv2d
        padded = F.pad(clean.unsqueeze(0), [self.pad] * 4, mode='reflect')
        blurred = F.conv2d(padded, self.kernel).squeeze(0)
        blurred = blurred.clamp(0.0, 1.0)
        return blurred, clean


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience loaders
# ═══════════════════════════════════════════════════════════════════════════════

def get_blur_task_loaders(task_id: int, batch_size: int = 128,
                          data_root: str = './data', num_workers: int = 4):
    """Return (train_loader, test_loader) for the given blur task."""
    train_ds = BlurredMNIST(task_id, train=True,  data_root=data_root)
    test_ds  = BlurredMNIST(task_id, train=False, data_root=data_root)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    return train_ld, test_ld


def task_name(task_id: int) -> str:
    return TASK_DEFS[task_id][0]
