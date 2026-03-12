"""
Task 1 & 2 — Train AlexNet-DCF (learnable atoms) on MNIST for each K.

For each value of K in K_LIST:
  - Build AlexNetDCF(num_bases=K, bases_grad=True, initializer='random')
  - Train for --epochs epochs
  - Record best test accuracy and number of parameters

Outputs saved to results/:
    dcf_results.json    — {K: best_test_acc, num_params, num_conv_params}

Usage:
    python train_dcf_alexnet.py [--epochs 30] [--lr 0.01] [--batch_size 128]
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
from models.alexnet_dcf import AlexNetDCF

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# K values: for 3×3 kernels, 9 = full rank.
# Also include K=9 for reference (equivalent to standard conv in expressivity).
K_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9]


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


def train_for_K(K, args, device, train_ld, test_ld):
    set_seed(args.seed)
    # DCF with random-initialised learnable atoms
    model = AlexNetDCF(num_bases=K, bases_grad=True,
                       initializer='random', num_classes=10).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    conv_params  = model.count_conv_params()
    print(f'  [K={K}] total_params={total_params:,}  conv_params={conv_params:,}')

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
    print(f'Device: {device}\n')
    train_ld, test_ld = get_loaders(args.batch_size)

    results = {'K_list': [], 'best_test_acc': [], 'total_params': [], 'conv_params': []}

    for K in K_LIST:
        print(f'\n=== Training DCF-AlexNet with K={K} ===')
        best_acc, total_p, conv_p = train_for_K(K, args, device, train_ld, test_ld)
        results['K_list'].append(K)
        results['best_test_acc'].append(best_acc)
        results['total_params'].append(total_p)
        results['conv_params'].append(conv_p)
        print(f'  → best_test_acc={best_acc:.2f}%')

    out_path = os.path.join(RESULTS_DIR, 'dcf_results.json')
    with open(out_path, 'w') as f:
        json.dump({'args': vars(args), 'results': results}, f, indent=2)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
