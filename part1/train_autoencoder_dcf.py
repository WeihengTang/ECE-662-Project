"""
Task 3 — Train DCF / FB Denoising Autoencoders on MNIST, sweep K.

For each K in the sweep lists:
  - DCF:  DenoisingAutoencoderDCF(num_bases=K, bases_grad=True,  initializer='random')
  - FB:   DenoisingAutoencoderDCF(num_bases=K, bases_grad=False, initializer='FB')

Metric: PSNR between reconstructed clean image and ground-truth clean image.

Outputs saved to results/:
    autoencoder_dcf_results.json
    autoencoder_fb_results.json

Usage:
    python train_autoencoder_dcf.py [--epochs 30] [--lr 1e-3] [--batch_size 128]
                                    [--noise_std 0.3] [--gpu 0] [--seed 42]
                                    [--mode dcf|fb|both]
"""
import os, sys, json, argparse, random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.autoencoder_dcf import DenoisingAutoencoderDCF

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

K_LIST_DCF = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # DCF: any K up to 9 (full rank)
K_LIST_FB  = [1, 2, 3, 4, 5, 6]            # FB:  max 6 for kernel_size=3


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',     type=int,   default=30)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--batch_size', type=int,   default=128)
    p.add_argument('--noise_std',  type=float, default=0.3)
    p.add_argument('--gpu',        type=int,   default=0)
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--mode',       type=str,   default='both',
                   choices=['dcf', 'fb', 'both'])
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_loaders(batch_size):
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
    return (clean + torch.randn_like(clean) * std).clamp(0.0, 1.0)


def mse_to_psnr(mse_per_pixel: float) -> float:
    if mse_per_pixel == 0:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse_per_pixel)


def run_epoch(model, loader, criterion, optimizer, noise_std, device, train=True):
    model.train(train)
    total_mse, n_images = 0.0, 0
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
            # sum of squared errors over all pixels in batch
            total_mse += nn.functional.mse_loss(pred, clean, reduction='sum').item()
            n_images  += clean.size(0)
    # MSE per pixel (averaged over images and spatial dims)
    pixels_per_image = clean[0].numel()
    avg_mse_per_pixel = total_mse / (n_images * pixels_per_image)
    return mse_to_psnr(avg_mse_per_pixel)


def train_for_K(K, bases_grad, initializer, args, device, train_ld, test_ld):
    set_seed(args.seed)
    model = DenoisingAutoencoderDCF(
        num_bases=K, bases_grad=bases_grad, initializer=initializer
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_psnr = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_psnr = run_epoch(model, train_ld, criterion, optimizer,
                            args.noise_std, device, train=True)
        te_psnr = run_epoch(model, test_ld,  criterion, None,
                            args.noise_std, device, train=False)
        scheduler.step()
        if te_psnr > best_psnr:
            best_psnr = te_psnr
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f'    ep{epoch:3d}: train_psnr={tr_psnr:.2f}dB  '
                  f'test_psnr={te_psnr:.2f}dB  best={best_psnr:.2f}dB')
    return best_psnr, total_params


def run_sweep(K_list, bases_grad, initializer, label, args, device, train_ld, test_ld):
    results = {'K_list': [], 'best_psnr': [], 'total_params': []}
    for K in K_list:
        print(f'\n=== [{label}] K={K} ===')
        psnr, n_params = train_for_K(K, bases_grad, initializer,
                                     args, device, train_ld, test_ld)
        results['K_list'].append(K)
        results['best_psnr'].append(psnr)
        results['total_params'].append(n_params)
        print(f'  → best_test_psnr={psnr:.2f} dB  total_params={n_params:,}')
    return results


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')
    train_ld, test_ld = get_loaders(args.batch_size)

    if args.mode in ('dcf', 'both'):
        print('===== DCF Autoencoder Sweep =====')
        dcf_res = run_sweep(K_LIST_DCF, True, 'random', 'DCF',
                            args, device, train_ld, test_ld)
        with open(os.path.join(RESULTS_DIR, 'autoencoder_dcf_results.json'), 'w') as f:
            json.dump({'args': vars(args), 'results': dcf_res}, f, indent=2)
        print('DCF results saved.')

    if args.mode in ('fb', 'both'):
        print('\n===== FB Autoencoder Sweep =====')
        fb_res = run_sweep(K_LIST_FB, False, 'FB', 'FB',
                           args, device, train_ld, test_ld)
        with open(os.path.join(RESULTS_DIR, 'autoencoder_fb_results.json'), 'w') as f:
            json.dump({'args': vars(args), 'results': fb_res}, f, indent=2)
        print('FB results saved.')


if __name__ == '__main__':
    main()
