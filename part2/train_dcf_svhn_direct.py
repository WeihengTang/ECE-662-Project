"""
Part 2 — Task 2, Experiment A: DCF-AlexNet trained directly on SVHN.

This gives the UPPER BOUND accuracy — what the DCF model can achieve when
it has access to full SVHN supervision (no cross-domain transfer required).

For each K in K_LIST:
  - Build AlexNetDCF(num_bases=K, bases_grad=True, initializer='random')
  - Train on SVHN (1-channel grayscale via channel averaging)
  - Record best test accuracy and parameter count

Outputs (saved to results_part2/):
    svhn_direct_results.json    — K_list, best_test_acc, total_params, conv_params

Usage:
    python train_dcf_svhn_direct.py [--epochs 30] [--lr 0.01] [--batch_size 128]
                                    [--gpu 0] [--seed 42]
"""
import os, sys, json, argparse, random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'part1'))

from config import RESULTS_DIR
from datasets_part2 import get_svhn_loaders
from models.alexnet_dcf import AlexNetDCF

K_LIST = [1, 2, 4, 6, 8]


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
    model = AlexNetDCF(num_bases=K, bases_grad=True,
                       initializer='random', num_classes=10).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    conv_params  = model.count_conv_params()
    print(f'  [K={K}] total={total_params:,}  conv_params={conv_params:,}')

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

    svhn_train, svhn_test = get_svhn_loaders(args.batch_size)
    results = {'K_list': [], 'best_test_acc': [], 'total_params': [], 'conv_params': []}

    for K in K_LIST:
        print(f'\n=== Task 2A — DCF-AlexNet trained on SVHN, K={K} ===')
        best_acc, total_p, conv_p = train_for_K(K, args, device, svhn_train, svhn_test)
        results['K_list'].append(K)
        results['best_test_acc'].append(best_acc)
        results['total_params'].append(total_p)
        results['conv_params'].append(conv_p)
        print(f'  → best_svhn_test_acc={best_acc:.2f}%')

    out_path = os.path.join(RESULTS_DIR, 'svhn_direct_results.json')
    with open(out_path, 'w') as f:
        json.dump({'args': vars(args), 'results': results}, f, indent=2)
    print(f'\nResults saved to {out_path}')


if __name__ == '__main__':
    main()
