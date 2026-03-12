"""
Part 2 — Task 2, Experiments B & C: Two-Stage DCF Adaptation.

Stage 1 (shared for both B and C):
    Train DCF-AlexNet on MNIST with BOTH atoms (ψ_k) and coefficients (a_k)
    as learnable parameters (bases_grad=True, initializer='random').
    Save the MNIST-trained checkpoint.

Stage 2B — "shared a_k, domain-specific ψ_k" (paper's approach):
    Load Stage-1 checkpoint.
    FREEZE  : a_k (self.weight) — shared semantic structure
    UNFREEZE: ψ_k (self.bases)  — adapt domain-specific low-level features
    Fine-tune on SVHN, evaluate test accuracy.

Stage 2C — Ablation "shared ψ_k, domain-specific a_k":
    Load Stage-1 checkpoint.
    FREEZE  : ψ_k (self.bases)  — shared structural atoms
    UNFREEZE: a_k (self.weight) — re-learn combination weights for SVHN
    Fine-tune on SVHN, evaluate test accuracy.

For each K in K_LIST, all three stages run sequentially.

Atoms (ψ_k) from Conv1 are saved after Stage 1 and Stage 2B for visualisation.

Outputs (saved to results_part2/):
    adaptation_results.json        — K_list, expB_acc, expC_acc, pre_adapt_acc
    atoms_mnist_K{K}.npy           — Conv1 atoms after Stage 1 (MNIST)
    atoms_svhn_expB_K{K}.npy       — Conv1 atoms after Stage 2B (SVHN adapted)

Usage:
    python train_dcf_adaptation.py [--epochs_s1 30] [--epochs_s2 15]
                                   [--lr_s1 0.01] [--lr_s2 0.001]
                                   [--batch_size 128] [--gpu 0] [--seed 42]
"""
import os, sys, json, argparse, random, copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'part1'))

from config import RESULTS_DIR
from datasets_part2 import get_mnist_loaders, get_svhn_loaders, get_svhn_test_loader
from models.alexnet_dcf import AlexNetDCF
from adapt_utils import (
    freeze_coefficients_unfreeze_atoms,
    freeze_atoms_unfreeze_coefficients,
    extract_atoms,
    count_trainable_params,
)

K_LIST = [1, 2, 4, 6, 8]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs_s1',  type=int,   default=30,   help='Stage-1 (MNIST) epochs')
    p.add_argument('--epochs_s2',  type=int,   default=15,   help='Stage-2 (SVHN adapt) epochs')
    p.add_argument('--lr_s1',      type=float, default=0.01, help='Stage-1 LR')
    p.add_argument('--lr_s2',      type=float, default=1e-3, help='Stage-2 adapt LR')
    p.add_argument('--batch_size', type=int,   default=128)
    p.add_argument('--gpu',        type=int,   default=0)
    p.add_argument('--seed',       type=int,   default=42)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += out.argmax(1).eq(labels).sum().item()
        total    += imgs.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += model(imgs).argmax(1).eq(labels).sum().item()
        total   += imgs.size(0)
    return 100.0 * correct / total


# ── Stage 1: Train on MNIST ──────────────────────────────────────────────────

def stage1_train_mnist(K, args, device, mnist_train, mnist_test):
    """Train DCF-AlexNet on MNIST. Return the trained model."""
    set_seed(args.seed)
    model = AlexNetDCF(num_bases=K, bases_grad=True,
                       initializer='random', num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr_s1,
                          momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_s1)

    best_acc, best_state = 0.0, None
    for epoch in range(1, args.epochs_s1 + 1):
        tr_acc = train_one_epoch(model, mnist_train, criterion, optimizer, device)
        te_acc = evaluate(model, mnist_test, device)
        scheduler.step()
        if te_acc > best_acc:
            best_acc  = te_acc
            best_state = copy.deepcopy(model.state_dict())
        if epoch % 5 == 0 or epoch == args.epochs_s1:
            print(f'    [S1 ep{epoch:3d}] mnist_train={tr_acc:.2f}%  '
                  f'mnist_test={te_acc:.2f}%  best={best_acc:.2f}%')

    # Restore best MNIST weights
    model.load_state_dict(best_state)
    ckpt_path = os.path.join(RESULTS_DIR, f'dcf_mnist_K{K}.pth')
    torch.save(best_state, ckpt_path)
    print(f'  Stage-1 best MNIST acc={best_acc:.2f}%  checkpoint → {ckpt_path}')
    return model, best_acc


# ── Stage 2: Adapt to SVHN ───────────────────────────────────────────────────

def stage2_adapt(model_src, freeze_fn_name, args, device,
                 svhn_train, svhn_test, label):
    """
    Generic adaptation stage.

    Args:
        model_src    : MNIST-trained DCF-AlexNet (will be deep-copied)
        freeze_fn_name: 'expB' (freeze coefficients) or 'expC' (freeze atoms)
        label        : display label for logging
    """
    model = copy.deepcopy(model_src)

    if freeze_fn_name == 'expB':
        freeze_coefficients_unfreeze_atoms(model)
    elif freeze_fn_name == 'expC':
        freeze_atoms_unfreeze_coefficients(model)
    else:
        raise ValueError(f"Unknown freeze mode: {freeze_fn_name}")

    trainable = count_trainable_params(model)
    print(f'  [{label}] trainable params after freeze: {trainable:,}')

    criterion = nn.CrossEntropyLoss()
    # Only pass parameters with requires_grad=True to the optimiser
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_s2, momentum=0.9, weight_decay=5e-4, nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_s2)

    best_acc, best_state = 0.0, None
    for epoch in range(1, args.epochs_s2 + 1):
        tr_acc = train_one_epoch(model, svhn_train, criterion, optimizer, device)
        te_acc = evaluate(model, svhn_test, device)
        scheduler.step()
        if te_acc > best_acc:
            best_acc  = te_acc
            best_state = copy.deepcopy(model.state_dict())
        if epoch % 3 == 0 or epoch == args.epochs_s2:
            print(f'    [{label} ep{epoch:3d}] svhn_train={tr_acc:.2f}%  '
                  f'svhn_test={te_acc:.2f}%  best={best_acc:.2f}%')

    model.load_state_dict(best_state)
    return model, best_acc


# ── Main sweep ────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    mnist_train, mnist_test = get_mnist_loaders(args.batch_size)
    svhn_train,  svhn_test  = get_svhn_loaders(args.batch_size)

    results = {
        'K_list':        [],
        'mnist_acc_s1':  [],     # MNIST test acc after Stage 1
        'pre_adapt_svhn_acc': [], # SVHN acc using Stage-1 model (no adaptation)
        'expB_svhn_acc': [],     # SVHN acc after Stage-2B (shared coefficients)
        'expC_svhn_acc': [],     # SVHN acc after Stage-2C (shared atoms)
    }

    for K in K_LIST:
        print(f'\n{"="*60}')
        print(f'  K = {K}')
        print(f'{"="*60}')

        # ── Stage 1: Train on MNIST ──────────────────────────────────────────
        print('\n--- Stage 1: Train DCF-AlexNet on MNIST ---')
        mnist_model, mnist_acc = stage1_train_mnist(
            K, args, device, mnist_train, mnist_test
        )

        # Save Conv1 atoms (MNIST domain)
        atoms_mnist = extract_atoms(mnist_model, layer_idx=0)
        np.save(os.path.join(RESULTS_DIR, f'atoms_mnist_K{K}.npy'), atoms_mnist)

        # Pre-adaptation SVHN accuracy (MNIST model → SVHN, no fine-tuning)
        pre_acc = evaluate(mnist_model, svhn_test, device)
        print(f'  Pre-adaptation SVHN acc (no fine-tune): {pre_acc:.2f}%')

        # ── Stage 2B: Freeze a_k, update ψ_k ───────────────────────────────
        print('\n--- Stage 2B: Adapt atoms (ψ_k) to SVHN [freeze a_k] ---')
        model_expB, acc_expB = stage2_adapt(
            mnist_model, 'expB', args, device, svhn_train, svhn_test, 'ExpB'
        )
        # Save adapted Conv1 atoms (SVHN domain, Exp B)
        atoms_svhn_expB = extract_atoms(model_expB, layer_idx=0)
        np.save(os.path.join(RESULTS_DIR, f'atoms_svhn_expB_K{K}.npy'), atoms_svhn_expB)
        torch.save(model_expB.state_dict(),
                   os.path.join(RESULTS_DIR, f'dcf_svhn_expB_K{K}.pth'))

        # ── Stage 2C: Freeze ψ_k, update a_k ───────────────────────────────
        print('\n--- Stage 2C: Adapt coefficients (a_k) to SVHN [freeze ψ_k] ---')
        model_expC, acc_expC = stage2_adapt(
            mnist_model, 'expC', args, device, svhn_train, svhn_test, 'ExpC'
        )
        torch.save(model_expC.state_dict(),
                   os.path.join(RESULTS_DIR, f'dcf_svhn_expC_K{K}.pth'))

        # ── Record results ───────────────────────────────────────────────────
        results['K_list'].append(K)
        results['mnist_acc_s1'].append(mnist_acc)
        results['pre_adapt_svhn_acc'].append(pre_acc)
        results['expB_svhn_acc'].append(acc_expB)
        results['expC_svhn_acc'].append(acc_expC)

        print(f'\n  K={K} summary:')
        print(f'    MNIST test (S1):            {mnist_acc:.2f}%')
        print(f'    SVHN pre-adaptation:        {pre_acc:.2f}%')
        print(f'    SVHN ExpB (adapt atoms):    {acc_expB:.2f}%')
        print(f'    SVHN ExpC (adapt coeff):    {acc_expC:.2f}%')

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, 'adaptation_results.json')
    with open(out_path, 'w') as f:
        json.dump({'args': vars(args), 'results': results}, f, indent=2)
    print(f'\nAll adaptation results saved to {out_path}')


if __name__ == '__main__':
    main()
