"""
Fourier-Bessel (FB) basis computation.
Translated from the MATLAB code in the original DCFNet paper (ICML 2018).
Reference: https://github.com/ZeWang95/DCFNet-Pytorch
"""
import numpy as np
from scipy import special
from config import path_to_bessel


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return phi, rho


def calculate_FB_bases(L1):
    """
    Compute Fourier-Bessel bases for a kernel of radius L1.

    Args:
        L1 (int): Half-size of the kernel. For kernel_size k, L1 = (k-1)//2.
                  E.g., L1=1 gives 3×3 bases; L1=2 gives 5×5 bases.

    Returns:
        psi  (ndarray): shape (k*k, num_bases), the basis matrix (columns are atoms)
        c    (float):   normalisation constant
        kq_Psi (ndarray): angular/radial frequency metadata
    """
    maxK = (2 * L1 + 1) ** 2 - 1
    L = L1 + 1
    R = L1 + 0.5

    truncate_freq_factor = 1.5
    if L1 < 2:
        truncate_freq_factor = 2.0

    xx, yy = np.meshgrid(range(-L, L + 1), range(-L, L + 1))
    xx = xx / R
    yy = yy / R

    ugrid = np.concatenate([yy.reshape(-1, 1), xx.reshape(-1, 1)], axis=1)
    tgrid, rgrid = cart2pol(ugrid[:, 0], ugrid[:, 1])

    kmax = 15
    bessel = np.load(path_to_bessel)
    B = bessel[(bessel[:, 0] <= kmax) & (bessel[:, 3] <= np.pi * R * truncate_freq_factor)]

    idxB = np.argsort(B[:, 2])
    ang_freqs = B[idxB, 0]
    rad_freqs = B[idxB, 1]
    R_ns = B[idxB, 2]

    num_grid_points = ugrid.shape[0]
    Psi = []
    kq_Psi = []
    num_bases = 0

    for i in range(B.shape[0]):
        ki = ang_freqs[i]
        qi = rad_freqs[i]
        rkqi = R_ns[i]

        r0grid = rgrid * R_ns[i]
        F = special.jv(ki, r0grid)
        Phi = (1.0 / np.abs(special.jv(ki + 1, R_ns[i]))) * F
        Phi[rgrid >= 1] = 0

        if ki == 0:
            Psi.append(Phi)
            kq_Psi.append([ki, qi, rkqi])
            num_bases += 1
        else:
            Psi.append(Phi * np.cos(ki * tgrid) * np.sqrt(2))
            Psi.append(Phi * np.sin(ki * tgrid) * np.sqrt(2))
            kq_Psi.append([ki, qi, rkqi])
            kq_Psi.append([ki, qi, rkqi])
            num_bases += 2

    Psi = np.array(Psi)
    kq_Psi = np.array(kq_Psi)

    if Psi.shape[0] > maxK:
        Psi = Psi[:maxK]
        kq_Psi = kq_Psi[:maxK]

    num_bases = Psi.shape[0]
    # Reshape to (num_bases, 2L+1, 2L+1), trim border, reshape to (k*k, num_bases)
    p = Psi.reshape(num_bases, 2 * L + 1, 2 * L + 1).transpose(1, 2, 0)
    psi = p[1:-1, 1:-1, :]
    psi = psi.reshape((2 * L1 + 1) ** 2, num_bases)

    c = np.sqrt(np.sum(psi ** 2, axis=0).mean())
    psi = psi / c

    return psi, c, kq_Psi


def get_fb_bases_tensor(kernel_size, num_bases):
    """
    Return a torch.Tensor of FB bases shaped (num_bases, kernel_size, kernel_size).

    Args:
        kernel_size (int): Odd integer (3, 5, 7, …).
        num_bases (int): Number of FB bases to keep (≤ max available).

    Returns:
        torch.Tensor: shape (num_bases, kernel_size, kernel_size)
    """
    import torch
    assert kernel_size % 2 == 1, "kernel_size must be odd for FB initialisation"
    L1 = (kernel_size - 1) // 2
    psi, _, _ = calculate_FB_bases(L1)  # (k*k, total_bases)
    total = psi.shape[1]
    if num_bases > total:
        raise ValueError(
            f"Requested {num_bases} FB bases but only {total} available "
            f"for kernel_size={kernel_size}"
        )
    psi = psi[:, :num_bases]  # (k*k, num_bases)
    # Transpose → (num_bases, k*k) → (num_bases, k, k)
    psi = psi.T.reshape(num_bases, kernel_size, kernel_size)
    return torch.tensor(psi, dtype=torch.float32)


def max_fb_bases(kernel_size):
    """Return the maximum number of FB bases for a given kernel_size."""
    L1 = (kernel_size - 1) // 2
    psi, _, _ = calculate_FB_bases(L1)
    return psi.shape[1]
