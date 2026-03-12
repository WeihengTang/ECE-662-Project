"""
Continual-learning utilities for Part 3.

Core idea  (from Miao et al., ICLR 2022, adapted to deblurring):
    Shared atoms   ψ_k ∈ R^{l×l}    — frozen after Task 0 training
    Per-task coeff. a_k^{(t)} ∈ R^{c'×c}  — learned & stored for each task t
    At inference for task t: swap in a_k^{(t)}, get W^{(t)} = Σ_k a_k^{(t)} ψ_k

This module provides:
    - coefficient extraction / injection for Conv_DCF layers
    - PSNR computation
    - memory-footprint counting
"""
import sys, os, copy
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'part1'))
from dcf_layer import Conv_DCF


# ═══════════════════════════════════════════════════════════════════════════════
# Coefficient extraction / injection  (the "atom-swapping" mechanism)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_task_coefficients(model: nn.Module) -> dict:
    """
    Extract all task-specific parameters (coefficients a_k + biases) from
    every Conv_DCF layer, plus the full decoder and batch-norm state.

    Returns a serialisable dict keyed by module path.
    """
    state = {}
    for name, m in model.named_modules():
        if isinstance(m, Conv_DCF):
            state[name + '.weight'] = m.weight.detach().cpu().clone()
            if m.bias is not None:
                state[name + '.bias'] = m.bias.detach().cpu().clone()
    # Also save batch-norm running stats and decoder (they are task-coupled)
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.ConvTranspose2d)):
            for pname, p in m.named_parameters():
                state[f'{name}.{pname}'] = p.detach().cpu().clone()
            for bname, b in m.named_buffers():
                state[f'{name}.{bname}'] = b.detach().cpu().clone()
    return state


def inject_task_coefficients(model: nn.Module, state: dict) -> None:
    """
    Restore a previously saved task-specific state into the model.

    This "swaps in" the task's coefficients, batch-norm stats, and decoder
    weights so the model behaves as if it were trained on that task.
    """
    full_state = model.state_dict()
    for key, val in state.items():
        if key in full_state:
            full_state[key] = val
    model.load_state_dict(full_state, strict=True)


def count_coefficients_memory(state: dict) -> int:
    """Return total number of scalar parameters in a task-specific state."""
    return sum(v.numel() for v in state.values())


# ═══════════════════════════════════════════════════════════════════════════════
# Freeze / unfreeze for continual-learning training
# ═══════════════════════════════════════════════════════════════════════════════

def freeze_atoms(model: nn.Module) -> None:
    """Freeze ψ_k (shared atoms) across all Conv_DCF layers."""
    for m in model.modules():
        if isinstance(m, Conv_DCF):
            if isinstance(m.bases, nn.Parameter):
                m.bases.requires_grad_(False)


def unfreeze_task_specific(model: nn.Module) -> None:
    """
    Ensure all task-specific parameters (coefficients, BN, decoder) are
    unfrozen and ready for optimisation.
    """
    for m in model.modules():
        if isinstance(m, Conv_DCF):
            m.weight.requires_grad_(True)
            if m.bias is not None:
                m.bias.requires_grad_(True)
        elif isinstance(m, (nn.BatchNorm2d, nn.ConvTranspose2d)):
            for p in m.parameters():
                p.requires_grad_(True)


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze every parameter in the model."""
    for p in model.parameters():
        p.requires_grad_(True)


# ═══════════════════════════════════════════════════════════════════════════════
# PSNR computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_psnr(model: nn.Module, loader, device: torch.device) -> float:
    """
    Compute average PSNR (dB) between model output and clean target
    over the entire loader.

    Assumes loader yields (blurred, clean) pairs with pixels in [0, 1].
    """
    model.eval()
    total_mse, n_pixels = 0.0, 0
    with torch.no_grad():
        for blurred, clean in loader:
            blurred, clean = blurred.to(device), clean.to(device)
            pred = model(blurred).clamp(0.0, 1.0)
            mse_sum = ((pred - clean) ** 2).sum().item()
            total_mse += mse_sum
            n_pixels  += clean.numel()
    avg_mse = total_mse / n_pixels
    if avg_mse < 1e-12:
        return 60.0   # cap at 60 dB
    return 10.0 * np.log10(1.0 / avg_mse)


# ═══════════════════════════════════════════════════════════════════════════════
# Model parameter counting
# ═══════════════════════════════════════════════════════════════════════════════

def count_all_params(model: nn.Module) -> int:
    """Total parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def count_shared_atoms(model: nn.Module) -> int:
    """Count parameters in shared atoms (ψ_k) across all Conv_DCF layers."""
    total = 0
    for m in model.modules():
        if isinstance(m, Conv_DCF):
            total += m.bases.numel()
    return total
