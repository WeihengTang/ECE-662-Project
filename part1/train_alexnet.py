"""
Train baseline AlexNetMNIST on MNIST.

Outputs (saved to results/):
    alexnet_mnist_best.pth   — best model weights (state_dict)
    alexnet_mnist_log.json   — per-epoch train/test accuracy and loss

Usage:
    python train_alexnet.py [--epochs 30] [--lr 0.01] [--batch_size 128]
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
from models.alexnet_mnist import AlexNetMNIST

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


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
    # Resize 28×28 → 32×32 so the three MaxPool2d(2) operations produce clean shapes.
    tfm = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(root='./data', train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST(root='./data', train=False, download=True, transform=tfm)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=256, shuffle=False,
                          num_workers=4, pin_memory=True)
    return train_ld, test_ld


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        correct      += out.argmax(1).eq(labels).sum().item()
        total        += imgs.size(0)
    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        running_loss += loss.item() * imgs.size(0)
        correct      += out.argmax(1).eq(labels).sum().item()
        total        += imgs.size(0)
    return running_loss / total, 100.0 * correct / total


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_ld, test_ld = get_loaders(args.batch_size)
    model = AlexNetMNIST(num_classes=10).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {total_params:,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_acc, best_epoch = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_ld, criterion, optimizer, device)
        te_loss, te_acc = evaluate(model, test_ld, criterion, device)
        scheduler.step()

        log['train_loss'].append(tr_loss)
        log['train_acc'].append(tr_acc)
        log['test_loss'].append(te_loss)
        log['test_acc'].append(te_acc)

        if te_acc > best_acc:
            best_acc = te_acc
            best_epoch = epoch
            torch.save(model.state_dict(),
                       os.path.join(RESULTS_DIR, 'alexnet_mnist_best.pth'))

        print(f'Epoch {epoch:3d}/{args.epochs}  '
              f'train_acc={tr_acc:.2f}%  test_acc={te_acc:.2f}%  '
              f'(best={best_acc:.2f}% @ ep{best_epoch})')

    # Save log
    log_path = os.path.join(RESULTS_DIR, 'alexnet_mnist_log.json')
    with open(log_path, 'w') as f:
        json.dump({'args': vars(args), 'log': log,
                   'best_test_acc': best_acc, 'best_epoch': best_epoch}, f, indent=2)
    print(f'\nBest test accuracy: {best_acc:.2f}% (epoch {best_epoch})')
    print(f'Weights saved to: {RESULTS_DIR}/alexnet_mnist_best.pth')
    print(f'Log saved to:     {log_path}')


if __name__ == '__main__':
    main()
