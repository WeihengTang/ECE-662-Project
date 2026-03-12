"""
Task 2 — Train AlexNet-FB (fixed Fourier-Bessel atoms) on MNIST for each K.

For each K in K_LIST:
  - Build AlexNetFB(num_bases=K)  [FB atoms fixed, only coefficients learned]
  - Train for --epochs epochs
  - Record best test accuracy and parameter count

For kernel_size=3, the maximum number of FB bases is 8.
If you try K > 8, the script will print a warning and skip that K.

Outputs saved to results/:
    fb_results.json   — {K_list, best_test_acc, total_params, conv_params}
    fb_atoms_K8.npy   — the 8 FB atoms (8, 3, 3) for visualisation

Usage:
    python train_fb_alexnet.py [--epochs 30] [--lr 0.01] [--batch_size 128]
                               [--gpu 0] [--seed 42]
"""
import os, sys, json, argparse, random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.alexnet_fb import AlexNetFB
from fb import get_fb_bases_tensor, max_fb_bases

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# For k=3, max FB bases = 8.  Sweep K from 1 to 8.
K_LIST_FB = [1, 2, 3, 4, 5, 6]   # max FB bases for kernel_size=3 is 6


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',     type=int,   default=30)
    p.add_argument('--lr',         type=float, default=0.01)
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


def get_loaders(batch_size):
    tfm = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST('./data', train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST('./data', train=False, download=True, transform=tfm)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=4, pin_memory=True),
        DataLoader(test_ds,  batch_size=256, shuffle=False,
                   num_workers=4, pin_memory=True),
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += out.argmax(1).eq(labels).sum().item()
        total    += imgs.size(0)
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += model(imgs).argmax(1).eq(labels).sum().item()
        total   += imgs.size(0)
    return 100.0 * correct / total


def count_fb_conv_params(model):
    """FB: only coefficients are trainable (atoms are fixed buffers)."""
    total = 0
    for m in model.modules():
        if hasattr(m, 'weight') and hasattr(m, 'num_bases') and hasattr(m, 'bases'):
            # coefficient tensor
            total += m.weight.numel()
            if m.bias is not None:
                total += m.bias.numel()
    return total


def train_for_K(K, args, device, train_ld, test_ld):
    set_seed(args.seed)
    model = AlexNetFB(num_bases=K, num_classes=10).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    conv_params  = count_fb_conv_params(model)
    print(f'  [K={K}] total_params={total_params:,}  conv_params(coeff only)={conv_params:,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        _, tr_acc = train_one_epoch(model, train_ld, criterion, optimizer, device)
        te_acc    = evaluate(model, test_ld, device)
        scheduler.step()
        if te_acc > best_acc:
            best_acc = te_acc
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f'    ep{epoch:3d}: train={tr_acc:.2f}%  test={te_acc:.2f}%  best={best_acc:.2f}%')

    return best_acc, total_params, conv_params


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Max FB bases for kernel_size=3: {max_fb_bases(3)}\n')

    train_ld, test_ld = get_loaders(args.batch_size)

    # Save FB atom visualisation (use max available bases for kernel_size=3)
    _max_k = max_fb_bases(3)
    fb_atoms = get_fb_bases_tensor(3, _max_k).numpy()
    np.save(os.path.join(RESULTS_DIR, f'fb_atoms_K{_max_k}.npy'), fb_atoms)
    print(f'FB atoms ({_max_k}×3×3) saved to results/fb_atoms_K{_max_k}.npy')

    results = {'K_list': [], 'best_test_acc': [], 'total_params': [], 'conv_params': []}

    for K in K_LIST_FB:
        print(f'\n=== Training FB-AlexNet with K={K} ===')
        best_acc, total_p, conv_p = train_for_K(K, args, device, train_ld, test_ld)
        results['K_list'].append(K)
        results['best_test_acc'].append(best_acc)
        results['total_params'].append(total_p)
        results['conv_params'].append(conv_p)
        print(f'  → best_test_acc={best_acc:.2f}%')

    out_path = os.path.join(RESULTS_DIR, 'fb_results.json')
    with open(out_path, 'w') as f:
        json.dump({'args': vars(args), 'results': results}, f, indent=2)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
