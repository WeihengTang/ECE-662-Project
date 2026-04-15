"""
Split-MNIST Deblurring Dataset for Continual Learning.

Splits MNIST into 5 sequential tasks (2 digits each):
    Task 0: digits {0, 1}
    Task 1: digits {2, 3}
    Task 2: digits {4, 5}
    Task 3: digits {6, 7}
    Task 4: digits {8, 9}

All images are degraded with the same Gaussian blur (sigma=1.5, 7x7).
Returns (blurred, clean) pairs in [0, 1], 1x28x28.

Usage:
    from split_mnist_blur import get_split_task_loaders, NUM_TASKS
    train_ld, test_ld = get_split_task_loaders(task_id=0, batch_size=128)
"""
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


def task_name(task_id: int) -> str:
    d = DIGITS_PER_TASK[task_id]
    return f'{d[0]},{d[1]}'


# ═════════════════════════════════════════════════════════════════════════════
# Blur kernel (shared across all tasks)
# ═════════════════════════════════════════════════════════════════════════════

def _gaussian_kernel(size: int = 7, sigma: float = 1.5) -> torch.Tensor:
    """Isotropic 2-D Gaussian kernel, normalised to sum 1."""
    coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g.outer(g)
    return kernel / kernel.sum()


BLUR_KERNEL = _gaussian_kernel(size=7, sigma=1.5)   # (7, 7)
BLUR_SIZE = 7
BLUR_PAD = (BLUR_SIZE - 1) // 2                     # 3


# ═════════════════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════════════════

class SplitMNISTBlur(Dataset):
    """
    Wraps torchvision MNIST, filters by digit class, and returns
    (blurred, clean) pairs.  Blur is applied on-the-fly (deterministic).
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
        self.kernel = BLUR_KERNEL.unsqueeze(0).unsqueeze(0)   # (1,1,7,7)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        clean, _label = self.mnist[real_idx]            # (1, 28, 28)
        padded = F.pad(clean.unsqueeze(0),
                       [BLUR_PAD] * 4, mode='reflect')
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
