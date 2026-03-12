"""
Part 2 — Task 1: Baseline (No Domain Adaptation)

Procedure:
  1. Train a standard AlexNetMNIST on MNIST for --epochs epochs.
  2. Evaluate the MNIST-trained model directly on SVHN (no fine-tuning).
  3. Report the domain-shift accuracy drop.

This establishes the lower bound that motivates DCF-based adaptation.

Outputs (saved to results_part2/):
    baseline_mnist_best.pth       — best AlexNet weights on MNIST val set
    baseline_results.json         — MNIST test acc + SVHN (no-adapt) test acc

Usage:
    python train_baseline_part2.py [--epochs 30] [--lr 0.01] [--batch_size 128]
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
from datasets_part2 import get_mnist_loaders, get_svhn_test_loader
from models.alexnet_mnist import AlexNetMNIST


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


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    mnist_train, mnist_test = get_mnist_loaders(args.batch_size)
    svhn_test               = get_svhn_test_loader()

    model     = AlexNetMNIST(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_mnist_acc, best_epoch = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        _, tr_acc  = train_one_epoch(model, mnist_train, criterion, optimizer, device)
        mnist_acc  = evaluate(model, mnist_test, device)
        scheduler.step()

        if mnist_acc > best_mnist_acc:
            best_mnist_acc = mnist_acc
            best_epoch     = epoch
            torch.save(model.state_dict(),
                       os.path.join(RESULTS_DIR, 'baseline_mnist_best.pth'))

        if epoch % 5 == 0 or epoch == args.epochs:
            print(f'Epoch {epoch:3d}/{args.epochs}  '
                  f'train_acc={tr_acc:.2f}%  mnist_test={mnist_acc:.2f}%  '
                  f'(best={best_mnist_acc:.2f}% @ ep{best_epoch})')

    # ── Evaluate best MNIST model on SVHN (no adaptation) ───────────────────
    print('\n[Baseline] Loading best MNIST model and evaluating on SVHN ...')
    best_state = torch.load(os.path.join(RESULTS_DIR, 'baseline_mnist_best.pth'),
                            map_location=device)
    model.load_state_dict(best_state)
    svhn_acc = evaluate(model, svhn_test, device)

    print(f'\n{"="*55}')
    print(f'  MNIST test accuracy (best):          {best_mnist_acc:.2f}%')
    print(f'  SVHN test accuracy (no adaptation):  {svhn_acc:.2f}%')
    print(f'  Domain-shift accuracy drop:          {best_mnist_acc - svhn_acc:.2f}%')
    print(f'{"="*55}')

    results = {
        'args':              vars(args),
        'best_mnist_acc':    best_mnist_acc,
        'best_epoch':        best_epoch,
        'svhn_no_adapt_acc': svhn_acc,
    }
    out_path = os.path.join(RESULTS_DIR, 'baseline_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {out_path}')


if __name__ == '__main__':
    main()
