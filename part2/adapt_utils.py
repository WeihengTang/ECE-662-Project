"""
Utility functions for DCF-based domain adaptation (Part 2, Task 2).

Two adaptation strategies are supported:

  Exp B — "shared coefficients, domain-specific atoms"  (the paper's approach):
      Freeze  : a_k (self.weight)   ← shared semantic structure
      Unfreeze: ψ_k (self.bases)    ← domain-specific low-level features
      Rationale: The coefficient matrix a_k encodes what the filter *detects*
                 (orientation, frequency, etc.) — this is shared across domains.
                 Only the atomic *appearance* (ψ_k) needs to shift with domain.

  Exp C — "shared atoms, domain-specific coefficients"  (ablation):
      Freeze  : ψ_k (self.bases)    ← shared structural bases
      Unfreeze: a_k (self.weight)   ← domain-specific combination weights
      Rationale (expected failure): Forcing different domains to share the
                 same bases but vary *how* they combine them tends to destroy
                 the semantic alignment, as each domain must re-learn which
                 atoms are relevant — equivalent to feature re-weighting
                 without structural reuse.
"""
import sys, os
import copy
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'part1'))
from dcf_layer import Conv_DCF


# ── Parameter freezing helpers ───────────────────────────────────────────────

def freeze_coefficients_unfreeze_atoms(model: torch.nn.Module) -> None:
    """
    Exp B setup: freeze a_k (weight), unfreeze ψ_k (bases).

    After this call, only bases (atoms) will receive gradients.
    Raises RuntimeError if any Conv_DCF layer has bases as a buffer
    (bases_grad=False) rather than a Parameter — train with bases_grad=True.
    """
    for m in model.modules():
        if isinstance(m, Conv_DCF):
            # Freeze coefficients
            m.weight.requires_grad_(False)
            if m.bias is not None:
                m.bias.requires_grad_(False)
            # Unfreeze atoms — must be a Parameter (bases_grad=True)
            if not isinstance(m.bases, torch.nn.Parameter):
                raise RuntimeError(
                    "Conv_DCF.bases is a buffer, not a Parameter. "
                    "Train the model with bases_grad=True."
                )
            m.bases.requires_grad_(True)


def freeze_atoms_unfreeze_coefficients(model: torch.nn.Module) -> None:
    """
    Exp C setup: freeze ψ_k (bases), unfreeze a_k (weight).

    After this call, only weights (coefficients) and biases will receive gradients.
    """
    for m in model.modules():
        if isinstance(m, Conv_DCF):
            # Unfreeze coefficients
            m.weight.requires_grad_(True)
            if m.bias is not None:
                m.bias.requires_grad_(True)
            # Freeze atoms
            if not isinstance(m.bases, torch.nn.Parameter):
                raise RuntimeError(
                    "Conv_DCF.bases is a buffer, not a Parameter. "
                    "Train the model with bases_grad=True."
                )
            m.bases.requires_grad_(False)


def unfreeze_all_dcf(model: torch.nn.Module) -> None:
    """Restore all Conv_DCF parameters to requires_grad=True."""
    for m in model.modules():
        if isinstance(m, Conv_DCF):
            m.weight.requires_grad_(True)
            if m.bias is not None:
                m.bias.requires_grad_(True)
            if isinstance(m.bases, torch.nn.Parameter):
                m.bases.requires_grad_(True)


# ── Atom extraction for visualisation ────────────────────────────────────────

def extract_atoms(model: torch.nn.Module, layer_idx: int = 0) -> np.ndarray:
    """
    Extract the filter atoms (ψ_k) from the layer_idx-th Conv_DCF layer.

    Returns:
        atoms: ndarray of shape (K, kH, kW)
    """
    dcf_layers = [m for m in model.modules() if isinstance(m, Conv_DCF)]
    if layer_idx >= len(dcf_layers):
        raise IndexError(f"layer_idx={layer_idx} but model has {len(dcf_layers)} DCF layers.")
    m = dcf_layers[layer_idx]
    if isinstance(m.bases, torch.nn.Parameter):
        return m.bases.detach().cpu().numpy()
    else:
        return m.bases.cpu().numpy()


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
