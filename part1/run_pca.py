"""
Task 1 — PCA on Trained AlexNet Filters.

Steps
-----
1. Load trained AlexNetMNIST from results/alexnet_mnist_best.pth.
2. For each convolutional layer, extract weight W ∈ R^{c'×c×l×l},
   reshape to W' ∈ R^{c'c × l²}, and apply truncated SVD.
3. For each K ∈ K_LIST reconstruct the filters using only the top-K
   singular vectors, replace the layer weights in the model, and
   evaluate test accuracy on MNIST (no retraining).
4. Count conv parameters vs K.
5. Save results to results/pca_results.json.

Output JSON schema:
{
  "K_list": [...],
  "test_acc": [...],
  "num_conv_params": [...],
  "layer_explained_variance": { "features.0": [...], ... }
}
"""
import os, sys, json, copy
import numpy as np

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.alexnet_mnist import AlexNetMNIST

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── K values to evaluate ────────────────────────────────────────────────────
# For 3×3 kernels, maximum rank = 9 (l² = 9).  K=9 → full reconstruction.
K_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def get_test_loader(batch_size=256):
    tfm = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    ds = datasets.MNIST('./data', train=False, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        correct += out.argmax(1).eq(labels).sum().item()
        total   += imgs.size(0)
    return 100.0 * correct / total


def pca_reconstruct_weights(W_np: np.ndarray, K: int):
    """
    Reconstruct weight tensor W using its top-K PCA components.

    Args:
        W_np : ndarray of shape (c_out, c_in, kH, kW)
        K    : number of components to keep

    Returns:
        W_rec : ndarray of same shape, K-rank approximation
        atoms : ndarray (K, kH, kW) — the filter atoms ψ_k
        coeffs: ndarray (c_out, c_in, K) — the coefficients a_k
    """
    c_out, c_in, kH, kW = W_np.shape
    W_flat = W_np.reshape(c_out * c_in, kH * kW)  # (c'c, l²)

    # SVD  — W_flat = U S Vt
    U, S, Vt = np.linalg.svd(W_flat, full_matrices=False)  # U:(c'c,r), S:(r,), Vt:(r,l²)

    # Truncate to K components
    K = min(K, S.shape[0])
    U_k  = U[:, :K]   # (c'c, K)
    S_k  = S[:K]      # (K,)
    Vt_k = Vt[:K, :]  # (K, l²)

    # Coefficients a_k[i,j] = U_k[i*c_in+j, k] * S_k[k]
    coeffs_flat = U_k * S_k[np.newaxis, :]  # (c'c, K)
    coeffs = coeffs_flat.reshape(c_out, c_in, K)

    # Atoms ψ_k = Vt_k[k].reshape(kH, kW)
    atoms = Vt_k.reshape(K, kH, kW)

    # Reconstruct: W_rec[i,j] = sum_k coeffs[i,j,k] * atoms[k]
    W_rec_flat = coeffs_flat @ Vt_k          # (c'c, l²)
    W_rec = W_rec_flat.reshape(c_out, c_in, kH, kW)

    return W_rec, atoms, coeffs


def count_conv_params_pca(model, K):
    """
    Count conv parameters if all conv layers use K-atom DCF decomposition.
    DCF conv(c_in, c_out, k=3, K):
        learnable atoms: K * k * k   (if bases_grad=True — DCF from scratch)
        coefficients:    c_out * c_in * K
    For the *PCA* case (post-hoc, no retraining), we count the same way
    as the DCF-from-scratch case to get a fair comparison.
    """
    total = 0
    for _, m in model.conv_layers:
        c_out, c_in, kH, kW = m.weight.shape
        total += K * kH * kW + c_out * c_in * K
    return total


def explained_variance_ratio(W_np: np.ndarray):
    """Return cumulative explained variance ratio (as list) for W_np."""
    W_flat = W_np.reshape(W_np.shape[0] * W_np.shape[1], -1)
    _, S, _ = np.linalg.svd(W_flat, full_matrices=False)
    evr = (S ** 2) / (S ** 2).sum()
    return np.cumsum(evr).tolist()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    weight_path = os.path.join(RESULTS_DIR, 'alexnet_mnist_best.pth')
    assert os.path.exists(weight_path), \
        f'Trained weights not found at {weight_path}. Run train_alexnet.py first.'

    # Load baseline model
    baseline = AlexNetMNIST(num_classes=10)
    baseline.load_state_dict(torch.load(weight_path, map_location='cpu'))
    baseline.to(device)

    test_loader = get_test_loader()

    # Baseline accuracy (K = all 9 components → lossless)
    baseline_acc = evaluate(baseline, test_loader, device)
    print(f'Baseline (full) accuracy: {baseline_acc:.2f}%')

    # Collect original conv weights
    original_weights = {}
    for name, m in baseline.conv_layers:
        original_weights[name] = m.weight.detach().cpu().numpy().copy()

    # Compute explained variance per layer
    layer_evr = {}
    for name, W_np in original_weights.items():
        layer_evr[name] = explained_variance_ratio(W_np)
        print(f'  {name}: EVR@9 = {layer_evr[name][-1]:.4f}')

    # ── Sweep K ─────────────────────────────────────────────────────────────
    test_accs      = []
    num_conv_params = []

    for K in K_LIST:
        # Deep-copy model to avoid in-place corruption
        model_k = copy.deepcopy(baseline)
        model_k.to(device)

        # Replace each conv layer's weight with its K-rank approximation
        with torch.no_grad():
            for name, m in model_k.conv_layers:
                W_np = original_weights[name]
                W_rec, _, _ = pca_reconstruct_weights(W_np, K)
                m.weight.copy_(torch.tensor(W_rec, dtype=torch.float32))

        acc_k = evaluate(model_k, test_loader, device)
        params_k = count_conv_params_pca(baseline, K)
        test_accs.append(acc_k)
        num_conv_params.append(params_k)
        print(f'  K={K:2d}  test_acc={acc_k:.2f}%  conv_params={params_k:,}')

        del model_k

    # Baseline conv params (all 9 components)
    baseline_conv_params = sum(
        m.weight.numel() for _, m in baseline.conv_layers
    )

    # ── Save atoms for visualisation ─────────────────────────────────────────
    # Save top-8 atoms for the first conv layer
    atoms_dict = {}
    W0 = original_weights[baseline.conv_layers[0][0]]
    _, atoms0, _ = pca_reconstruct_weights(W0, 8)
    atoms_dict['features_conv0_atoms'] = atoms0.tolist()  # (8, 3, 3)

    results = {
        'K_list': K_LIST,
        'test_acc': test_accs,
        'baseline_test_acc': baseline_acc,
        'num_conv_params': num_conv_params,
        'baseline_conv_params': baseline_conv_params,
        'layer_explained_variance': layer_evr,
        'atoms': atoms_dict,
    }

    out_path = os.path.join(RESULTS_DIR, 'pca_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
