"""
Split-MNIST Deblurring Dataset for Continual Learning.

Splits MNIST into 5 sequential tasks (2 digits each), each with a
DIFFERENT blur kernel so the deblurring functions genuinely conflict:

    Task 0: digits {0, 1}  — Gaussian blur  (sigma=1.5)
    Task 1: digits {2, 3}  — Motion blur    (45 deg)
    Task 2: digits {4, 5}  — Defocus blur   (radius=2.5)
    Task 3: digits {6, 7}  — Gaussian blur  (sigma=2.5)
    Task 4: digits {8, 9}  — Motion blur    (0 deg, horizontal)

Returns (blurred, clean) pairs in [0, 1], 1x28x28.

Usage:
    from split_mnist_blur import get_split_task_loaders, NUM_TASKS
    train_ld, test_ld = get_split_task_loaders(task_id=0, batch_size=128)
"""
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


# ═════════════════════════════════════════════════════════════════════════════
# Task definitions
# ═════════════════════════════════════════════════════════════════════════════

NUM_TASKS = 5

DIGITS_PER_TASK = [
    [0, 1],   # Task 0
    [2, 3],   # Task 1
    [4, 5],   # Task 2
    [6, 7],   # Task 3
    [8, 9],   # Task 4
]

BLUR_DESCRIPTIONS = [
    'Gaussian $\\sigma$=1.5',
    'Motion 45$^\\circ$',
    'Defocus R=2.5',
    'Gaussian $\\sigma$=2.5',
    'Motion 0$^\\circ$',
]


def task_name(task_id: int) -> str:
    d = DIGITS_PER_TASK[task_id]
    return f'{d[0]},{d[1]}'


def task_label(task_id: int) -> str:
    """Full task label including blur type (for plots)."""
    d = DIGITS_PER_TASK[task_id]
    blur_short = ['Gauss1.5', 'Mot45', 'Defocus', 'Gauss2.5', 'Mot0']
    return f'{{{d[0]},{d[1]}}} {blur_short[task_id]}'


# ═════════════════════════════════════════════════════════════════════════════
# Blur kernel constructors
# ═════════════════════════════════════════════════════════════════════════════

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
    dx, dy = math.cos(angle_rad), math.sin(angle_rad)
    length = (size - 1) / 2.0
    for t_i in range(200):
        t = -length + 2 * length * t_i / 199.0
        xi = int(round(centre + t * dx))
        yi = int(round(centre + t * dy))
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


# Per-task blur kernels: (kernel_tensor, kernel_size)
TASK_BLUR = [
    (_gaussian_kernel(7, sigma=1.5),     7),   # Task 0
    (_motion_kernel(7, angle_deg=45.0),  7),   # Task 1
    (_disk_kernel(7, radius=2.5),        7),   # Task 2
    (_gaussian_kernel(7, sigma=2.5),     7),   # Task 3
    (_motion_kernel(7, angle_deg=0.0),   7),   # Task 4
]


def get_blur_kernel(task_id: int) -> torch.Tensor:
    """Return blur kernel for a task as (1, 1, kH, kW) tensor."""
    kernel, _ = TASK_BLUR[task_id]
    return kernel.unsqueeze(0).unsqueeze(0)


# ═════════════════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════════════════

class SplitMNISTBlur(Dataset):
    """
    Wraps torchvision MNIST, filters by digit class, and returns
    (blurred, clean) pairs.  Each task has its own blur kernel.
    """

    def __init__(self, task_id: int, train: bool = True,
                 data_root: str = './data'):
        super().__init__()
        tfm = transforms.Compose([transforms.ToTensor()])
        full = datasets.MNIST(data_root, train=train, download=True,
                              transform=tfm)

        # Filter indices by digit class
        digits = set(DIGITS_PER_TASK[task_id])
        self.indices = [i for i in range(len(full))
                        if full.targets[i].item() in digits]
        self.mnist = full
        self.kernel = get_blur_kernel(task_id)                  # (1,1,kH,kW)
        self.pad = (self.kernel.shape[-1] - 1) // 2

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        clean, _label = self.mnist[real_idx]            # (1, 28, 28)
        padded = F.pad(clean.unsqueeze(0),
                       [self.pad] * 4, mode='reflect')
        blurred = F.conv2d(padded, self.kernel).squeeze(0)
        blurred = blurred.clamp(0.0, 1.0)
        return blurred, clean


# ═════════════════════════════════════════════════════════════════════════════
# Convenience loaders
# ═════════════════════════════════════════════════════════════════════════════

def get_split_task_loaders(task_id: int, batch_size: int = 128,
                           data_root: str = './data', num_workers: int = 4):
    """Return (train_loader, test_loader) for the given task."""
    train_ds = SplitMNISTBlur(task_id, train=True,  data_root=data_root)
    test_ds  = SplitMNISTBlur(task_id, train=False, data_root=data_root)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    return train_ld, test_ld
