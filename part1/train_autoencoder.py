"""
Task 3 — Train baseline Denoising Autoencoder on MNIST.

The autoencoder is trained to reconstruct a *clean* digit from a *noisy*
input (clean + Gaussian noise with std=0.3, clipped to [0,1]).

Loss : MSE(reconstruction, clean)
PSNR : 10 * log10(1^2 / MSE)  [pixel range [0,1], so MAX=1]

Outputs saved to results/:
    autoencoder_best.pth        — best model state_dict
    autoencoder_train_log.json  — per-epoch loss and PSNR

Usage:
    python train_autoencoder.py [--epochs 30] [--lr 1e-3] [--batch_size 128]
                                [--noise_std 0.3] [--gpu 0] [--seed 42]
"""
import os, sys, json, argparse, random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.autoencoder import DenoisingAutoencoder

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',     type=int,   default=30)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--batch_size', type=int,   default=128)
    p.add_argument('--noise_std',  type=float, default=0.3)
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
    # No normalisation — keep pixels in [0,1] for PSNR computation.
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST('./data', train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST('./data', train=False, download=True, transform=tfm)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=4, pin_memory=True),
        DataLoader(test_ds,  batch_size=256, shuffle=False,
                   num_workers=4, pin_memory=True),
    )


def add_noise(clean, std, device):
    noise  = torch.randn_like(clean) * std
    noisy  = (clean + noise).clamp(0.0, 1.0)
    return noisy


def mse_to_psnr(mse: float, max_val: float = 1.0) -> float:
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(max_val ** 2 / mse)


def run_epoch(model, loader, criterion, optimizer, noise_std, device, train=True):
    model.train(train)
    total_loss, total_mse, n = 0.0, 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for clean, _ in loader:
            clean = clean.to(device)
            noisy = add_noise(clean, noise_std, device)
            pred  = model(noisy)
            loss  = criterion(pred, clean)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            bs = clean.size(0)
            total_loss += loss.item() * bs
            total_mse  += nn.functional.mse_loss(pred, clean, reduction='sum').item()
            n += bs * clean.numel() // bs  # pixels per image × images
    avg_mse = total_mse / (n * clean[0].numel() if n else 1)
    # More precise: average MSE per image
    avg_mse = total_mse / n if n else 0.0
    # Actually compute as average MSE per pixel
    avg_mse_pix = total_mse / (n * clean[0].numel()) if n else 0.0
    psnr = mse_to_psnr(avg_mse_pix)
    return total_loss / n, psnr


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    train_ld, test_ld = get_loaders(args.batch_size)
    model = DenoisingAutoencoder().to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {total_params:,}')

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    log = {'train_loss': [], 'train_psnr': [], 'test_loss': [], 'test_psnr': []}
    best_psnr, best_epoch = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_psnr = run_epoch(model, train_ld, criterion, optimizer,
                                     args.noise_std, device, train=True)
        te_loss, te_psnr = run_epoch(model, test_ld,  criterion, None,
                                     args.noise_std, device, train=False)
        scheduler.step()

        log['train_loss'].append(tr_loss)
        log['train_psnr'].append(tr_psnr)
        log['test_loss'].append(te_loss)
        log['test_psnr'].append(te_psnr)

        if te_psnr > best_psnr:
            best_psnr  = te_psnr
            best_epoch = epoch
            torch.save(model.state_dict(),
                       os.path.join(RESULTS_DIR, 'autoencoder_best.pth'))

        print(f'Epoch {epoch:3d}/{args.epochs}  '
              f'train_psnr={tr_psnr:.2f}dB  test_psnr={te_psnr:.2f}dB  '
              f'(best={best_psnr:.2f}dB @ ep{best_epoch})')

    log_path = os.path.join(RESULTS_DIR, 'autoencoder_train_log.json')
    with open(log_path, 'w') as f:
        json.dump({'args': vars(args), 'log': log,
                   'best_test_psnr': best_psnr, 'best_epoch': best_epoch}, f, indent=2)
    print(f'\nBest test PSNR: {best_psnr:.2f} dB (epoch {best_epoch})')
    print(f'Saved weights to {RESULTS_DIR}/autoencoder_best.pth')
    print(f'Saved log to    {log_path}')


if __name__ == '__main__':
    main()
