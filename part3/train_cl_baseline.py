"""
Part 3 — Baseline: Naive Sequential Training (Catastrophic Forgetting).

A standard convolutional autoencoder (no DCF decomposition) is trained
sequentially on Task 0 → Task 1 → Task 2 without any continual-learning
mechanism.  After each task finishes, we evaluate PSNR on **all** tasks
seen so far to measure forgetting.

Workflow:
  1. Train on Task 0 for E epochs.
     Evaluate PSNR on {Task 0}.
  2. Continue training the *same* model on Task 1 for E epochs.
     Evaluate PSNR on {Task 0, Task 1}.
  3. Continue training the *same* model on Task 2 for E epochs.
     Evaluate PSNR on {Task 0, Task 1, Task 2}.

The PSNR matrix is saved as:
  psnr_after_task[t][j] = PSNR on task j after the model finishes training on task t.

Outputs (saved to results_part3/):
    baseline_cl_results.json    — full PSNR matrix + param counts
    baseline_model_after_t{0,1,2}.pth

Usage:
    python train_cl_baseline.py [--epochs_per_task 20] [--lr 1e-3]
                                [--batch_size 128] [--gpu 0] [--seed 42]
"""
import os, sys, json, argparse, random, copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'part1'))

from dataset_blur import get_blur_task_loaders, NUM_TASKS, task_name
from models.autoencoder import DenoisingAutoencoder

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results_part3')
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs_per_task', type=int,   default=20)
    p.add_argument('--lr',              type=float, default=1e-3)
    p.add_argument('--batch_size',      type=int,   default=128)
    p.add_argument('--gpu',             type=int,   default=0)
    p.add_argument('--seed',            type=int,   default=42)
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def compute_psnr(model, loader, device):
    model.eval()
    total_mse, n = 0.0, 0
    with torch.no_grad():
        for blurred, clean in loader:
            blurred, clean = blurred.to(device), clean.to(device)
            pred = model(blurred).clamp(0.0, 1.0)
            total_mse += ((pred - clean) ** 2).sum().item()
            n += clean.numel()
    mse = total_mse / n
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for blurred, clean in loader:
        blurred, clean = blurred.to(device), clean.to(device)
        pred = model(blurred)
        loss = criterion(pred, clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * clean.size(0)
        n += clean.size(0)
    return total_loss / n


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available()
                          else 'cpu')
    print(f'Device: {device}\n')

    model     = DenoisingAutoencoder().to(device)
    criterion = nn.MSELoss()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Baseline autoencoder params: {total_params:,}\n')

    # Pre-load all test loaders (cheap; used for evaluation after each task)
    test_loaders = {}
    for t in range(NUM_TASKS):
        _, tl = get_blur_task_loaders(t, args.batch_size)
        test_loaders[t] = tl

    # PSNR matrix: psnr_after[t][j] = PSNR on task j after training on task t
    psnr_after = {}

    for t in range(NUM_TASKS):
        print(f'{"="*55}')
        print(f' Training on Task {t}: {task_name(t)}')
        print(f'{"="*55}')

        train_ld, _ = get_blur_task_loaders(t, args.batch_size)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs_per_task)

        for ep in range(1, args.epochs_per_task + 1):
            loss = train_one_epoch(model, train_ld, criterion, optimizer, device)
            scheduler.step()
            if ep % 5 == 0 or ep == args.epochs_per_task:
                print(f'  ep {ep:3d}  loss={loss:.6f}')

        # Save model after this task
        torch.save(model.state_dict(),
                   os.path.join(RESULTS_DIR, f'baseline_model_after_t{t}.pth'))

        # Evaluate on all tasks seen so far (and even future ones for reference)
        psnr_row = {}
        for j in range(NUM_TASKS):
            p = compute_psnr(model, test_loaders[j], device)
            psnr_row[j] = p
        psnr_after[t] = psnr_row

        print(f'\n  PSNR after Task {t}:')
        for j in range(NUM_TASKS):
            tag = '  (future)' if j > t else ''
            print(f'    Task {j} ({task_name(j):>8s}): {psnr_row[j]:.2f} dB{tag}')

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f'\n{"="*55}')
    print('Forgetting summary (Baseline — naive sequential):')
    for t in range(NUM_TASKS):
        for j in range(t + 1):
            print(f'  After Task {t}, PSNR on Task {j}: '
                  f'{psnr_after[t][j]:.2f} dB')
    print(f'{"="*55}')

    results = {
        'args': vars(args),
        'total_params': total_params,
        'params_per_task': total_params,   # naive: full model stored per task
        'psnr_after': {str(k): {str(j): v for j, v in row.items()}
                       for k, row in psnr_after.items()},
    }
    out = os.path.join(RESULTS_DIR, 'baseline_cl_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
