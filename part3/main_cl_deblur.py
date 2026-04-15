#!/usr/bin/env python3
"""
Split-MNIST Deblurring via Filter Atom Swapping (Continual Learning).

Five class-incremental tasks (2 digits each), single Gaussian blur.

DCF-CL protocol:
    Task 0 : train ALL params (atoms + coefficients + decoder + BN).
    Task t>0: freeze {coefficients, decoder, BN}; warm-start atoms from
              Task 0; train ONLY atoms (162 scalars for K=6, 3x3).
    Eval on Task j: inject saved atoms^{(j)} -> exact parameter restore.

Baseline: standard Conv autoencoder, naive sequential training, no CL.

Outputs (saved to results_part3_v2/):
    cl_deblur_results.json            - full PSNR matrices + memory analysis
    fig_p3v2_blur_examples.pdf        - sample blurred/clean pairs per task
    fig_p3v2_forgetting_curves.pdf    - PSNR per task across training phases
    fig_p3v2_psnr_heatmaps.pdf        - 5x5 PSNR matrices (baseline vs ours)
    fig_p3v2_memory_footprint.pdf     - parameter bar chart

Usage:
    python main_cl_deblur.py [--epochs_per_task 30] [--gpu 0]
"""
import os, sys, json, argparse, random, math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ═══ paths ═══════════════════════════════════════════════════════════════════
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'part1'))
sys.path.insert(0, _HERE)

from dcf_layer import Conv_DCF
from models.autoencoder_dcf import DenoisingAutoencoderDCF
from models.autoencoder import DenoisingAutoencoder
from split_mnist_blur import (get_split_task_loaders, NUM_TASKS,
                              DIGITS_PER_TASK, task_name, BLUR_KERNEL)

RESULTS = os.path.join(_HERE, 'results_part3_v2')
os.makedirs(RESULTS, exist_ok=True)


# ═══ reproducibility ═════════════════════════════════════════════════════════
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ═══ utilities ═══════════════════════════════════════════════════════════════
def count_params(model):
    return sum(p.numel() for p in model.parameters())


def compute_psnr(model, loader, device):
    """Average PSNR (dB) on loader; images in [0,1]."""
    model.eval()
    mse_sum, npx = 0.0, 0
    with torch.no_grad():
        for blur, clean in loader:
            blur, clean = blur.to(device), clean.to(device)
            pred = model(blur).clamp(0.0, 1.0)
            mse_sum += ((pred - clean) ** 2).sum().item()
            npx += clean.numel()
    avg_mse = mse_sum / npx
    if avg_mse < 1e-12:
        return 60.0
    return 10.0 * np.log10(1.0 / avg_mse)


# ═══ atom-swapping helpers ═══════════════════════════════════════════════════
def extract_atoms(model):
    """Return dict of atom tensors from all Conv_DCF layers."""
    state = {}
    for name, m in model.named_modules():
        if isinstance(m, Conv_DCF):
            state[name + '.bases'] = m.bases.detach().cpu().clone()
    return state


def inject_atoms(model, state):
    """Load saved atoms back into the model."""
    for name, m in model.named_modules():
        if isinstance(m, Conv_DCF):
            key = name + '.bases'
            if key in state:
                m.bases.data.copy_(state[key].to(m.bases.device))


def count_atom_scalars(state):
    return sum(v.numel() for v in state.values())


def freeze_all_except_atoms(model):
    """Freeze every param, then unfreeze only Conv_DCF bases."""
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, Conv_DCF):
            m.bases.requires_grad_(True)


def freeze_bn(model):
    """Set all BatchNorm layers to eval mode (use Task 0 running stats)."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad_(True)


# ═══ training ════════════════════════════════════════════════════════════════
def train_epoch(model, loader, criterion, optimizer, device, bn_frozen=False):
    model.train()
    if bn_frozen:
        freeze_bn(model)
    loss_total, n = 0.0, 0
    for blur, clean in loader:
        blur, clean = blur.to(device), clean.to(device)
        pred = model(blur)
        loss = criterion(pred, clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += loss.item() * clean.size(0)
        n += clean.size(0)
    return loss_total / n


# ═══ plotting ════════════════════════════════════════════════════════════════
def _savefig(fig, name):
    fig.savefig(os.path.join(RESULTS, name), bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f'  Saved {name}')


def plot_blur_examples(test_loaders, device):
    """One sample per task: blurred (top) and clean (bottom)."""
    fig, axes = plt.subplots(2, NUM_TASKS, figsize=(2.4 * NUM_TASKS, 5))
    for t in range(NUM_TASKS):
        blur, clean = next(iter(test_loaders[t]))
        b_img = blur[0, 0].cpu().numpy()
        c_img = clean[0, 0].cpu().numpy()
        axes[0, t].imshow(b_img, cmap='gray', vmin=0, vmax=1)
        axes[0, t].set_title(f'T{t}: {{{task_name(t)}}}', fontsize=10)
        axes[0, t].axis('off')
        axes[1, t].imshow(c_img, cmap='gray', vmin=0, vmax=1)
        axes[1, t].axis('off')
    axes[0, 0].set_ylabel('Blurred', fontsize=10)
    axes[1, 0].set_ylabel('Clean', fontsize=10)
    fig.suptitle('Split-MNIST Deblurring Tasks (Gaussian blur, $\\sigma$=1.5)',
                 fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _savefig(fig, 'fig_p3v2_blur_examples.pdf')


def plot_forgetting_curves(psnr_dcf, psnr_base):
    """PSNR per task across training phases."""
    fig, axes = plt.subplots(1, NUM_TASKS, figsize=(3.2 * NUM_TASKS, 3.5),
                             sharey=True)
    phases = list(range(NUM_TASKS))
    for j in range(NUM_TASKS):
        ax = axes[j]
        # Baseline
        vals_b = [psnr_base[t][j] for t in phases]
        ax.plot(phases, vals_b, 'r--o', ms=4, label='Baseline')
        # DCF-CL
        vals_d = [psnr_dcf[t][j] for t in phases]
        ax.plot(phases, vals_d, 'b-s', ms=4, label='DCF-CL')
        ax.set_title(f'Task {j}  ({{{task_name(j)}}})', fontsize=10)
        ax.set_xlabel('Training phase')
        ax.set_xticks(phases)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel('PSNR (dB)')
            ax.legend(fontsize=8)
    fig.suptitle('Forgetting Curves: PSNR per Task over Training Phases',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    _savefig(fig, 'fig_p3v2_forgetting_curves.pdf')


def plot_psnr_heatmaps(psnr_dcf, psnr_base):
    """Side-by-side 5x5 PSNR heatmaps."""
    def _build_matrix(psnr_dict):
        mat = np.zeros((NUM_TASKS, NUM_TASKS))
        for t in range(NUM_TASKS):
            for j in range(NUM_TASKS):
                mat[t, j] = psnr_dict[t][j]
        return mat

    mat_b = _build_matrix(psnr_base)
    mat_d = _build_matrix(psnr_dcf)

    vmin = min(mat_b.min(), mat_d.min()) - 1
    vmax = max(mat_b.max(), mat_d.max()) + 1
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, mat, title in [(ax1, mat_b, 'Baseline (naive)'),
                           (ax2, mat_d, 'DCF-CL (atom swap)')]:
        im = ax.imshow(mat, cmap='YlOrRd', norm=norm, aspect='equal')
        for i in range(NUM_TASKS):
            for j in range(NUM_TASKS):
                ax.text(j, i, f'{mat[i,j]:.1f}', ha='center', va='center',
                        fontsize=8, color='black')
        ax.set_xticks(range(NUM_TASKS))
        ax.set_xticklabels([f'T{j}' for j in range(NUM_TASKS)])
        ax.set_yticks(range(NUM_TASKS))
        ax.set_yticklabels([f'After T{t}' for t in range(NUM_TASKS)])
        ax.set_xlabel('Evaluation task')
        ax.set_ylabel('Training phase')
        ax.set_title(title)
    fig.colorbar(im, ax=[ax1, ax2], label='PSNR (dB)', shrink=0.8)
    fig.tight_layout()
    _savefig(fig, 'fig_p3v2_psnr_heatmaps.pdf')


def plot_memory_footprint(shared, per_task, baseline_per_model):
    """Stacked bar: baseline vs DCF-CL."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    base_total = NUM_TASKS * baseline_per_model
    dcf_total = shared + NUM_TASKS * per_task

    # Baseline: solid bar
    ax.bar('Baseline\n(naive)', base_total, color='#d9534f',
           label=f'{NUM_TASKS} full models')

    # DCF-CL: stacked
    ax.bar('DCF-CL\n(atom swap)', shared, color='#5bc0de',
           label='Shared (coeffs+decoder+BN)')
    ax.bar('DCF-CL\n(atom swap)', NUM_TASKS * per_task,
           bottom=shared, color='#f0ad4e',
           label=f'{NUM_TASKS} x atoms ({per_task} each)')

    ax.set_ylabel('Total parameters stored')
    ax.set_title(f'Memory Footprint ({NUM_TASKS} tasks)')

    # Annotation
    ratio = base_total / dcf_total
    ax.text(1, dcf_total + base_total * 0.03,
            f'{ratio:.1f}x compression', ha='center', fontsize=10,
            fontweight='bold')

    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, base_total * 1.25)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    fig.tight_layout()
    _savefig(fig, 'fig_p3v2_memory_footprint.pdf')


# ═══ main ════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description='Split-MNIST CL Deblurring')
    ap.add_argument('--num_bases',       type=int,   default=6)
    ap.add_argument('--epochs_per_task', type=int,   default=30)
    ap.add_argument('--lr',              type=float, default=1e-3)
    ap.add_argument('--batch_size',      type=int,   default=128)
    ap.add_argument('--gpu',             type=int,   default=0)
    ap.add_argument('--seed',            type=int,   default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available()
                          else 'cpu')
    print(f'Device: {device}\n')

    # Pre-load ALL test loaders (used for cross-task evaluation)
    test_loaders = {}
    for t in range(NUM_TASKS):
        _, tl = get_split_task_loaders(t, args.batch_size)
        test_loaders[t] = tl

    criterion = nn.MSELoss()

    # ════════════════════════════════════════════════════════════════════════
    #  A. DCF-CL — Filter Atom Swapping
    # ════════════════════════════════════════════════════════════════════════
    print('=' * 60)
    print('  DCF-CL: Filter Atom Swapping')
    print('=' * 60)

    dcf = DenoisingAutoencoderDCF(
        num_bases=args.num_bases, bases_grad=True, initializer='FB'
    ).to(device)

    dcf_total_params = count_params(dcf)
    print(f'DCF model total params: {dcf_total_params:,}')

    task_atoms = {}
    psnr_dcf = {}       # psnr_dcf[t][j] = PSNR on task j after training task t

    for t in range(NUM_TASKS):
        print(f'\n{"─"*50}')
        print(f' Task {t}: digits {{{task_name(t)}}}')
        print(f'{"─"*50}')

        train_ld, _ = get_split_task_loaders(t, args.batch_size)

        if t == 0:
            # ── Task 0: train everything ────────────────────────────────
            unfreeze_all(dcf)
            trainable = sum(p.numel() for p in dcf.parameters()
                            if p.requires_grad)
            print(f'  Training ALL params ({trainable:,})')
            bn_frozen = False
        else:
            # ── Task t>0: warm-start atoms from Task 0, freeze rest ────
            inject_atoms(dcf, task_atoms[0])      # warm start
            freeze_all_except_atoms(dcf)
            trainable = sum(p.numel() for p in dcf.parameters()
                            if p.requires_grad)
            print(f'  Warm-start atoms from Task 0, training ONLY atoms '
                  f'({trainable:,} params)')
            bn_frozen = True

        opt = optim.Adam([p for p in dcf.parameters() if p.requires_grad],
                         lr=args.lr)
        sched = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs_per_task)

        for ep in range(1, args.epochs_per_task + 1):
            loss = train_epoch(dcf, train_ld, criterion, opt, device,
                               bn_frozen)
            sched.step()
            if ep % 10 == 0 or ep == args.epochs_per_task:
                print(f'    ep {ep:3d}  loss={loss:.6f}')

        # Save atoms for this task
        task_atoms[t] = extract_atoms(dcf)
        n_atoms = count_atom_scalars(task_atoms[t])
        torch.save(task_atoms[t],
                   os.path.join(RESULTS, f'dcf_atoms_task{t}.pth'))
        print(f'  Saved Task-{t} atoms: {n_atoms:,} scalars')

        # ── Evaluate on ALL tasks (swap atoms) ─────────────────────────
        psnr_dcf[t] = {}
        for j in range(NUM_TASKS):
            if j <= t:
                inject_atoms(dcf, task_atoms[j])
            else:
                # Unseen task: use most recently trained atoms
                inject_atoms(dcf, task_atoms[t])
            psnr_dcf[t][j] = compute_psnr(dcf, test_loaders[j], device)

        # Restore current task atoms
        inject_atoms(dcf, task_atoms[t])

        print(f'  PSNR after Task {t}:')
        for j in range(NUM_TASKS):
            tag = ' (unseen)' if j > t else ''
            print(f'    T{j} ({task_name(j):>3s}): '
                  f'{psnr_dcf[t][j]:.2f} dB{tag}')

    # ════════════════════════════════════════════════════════════════════════
    #  B. Baseline — Naive Sequential Training
    # ════════════════════════════════════════════════════════════════════════
    print(f'\n{"=" * 60}')
    print('  Baseline: Naive Sequential Conv Autoencoder')
    print('=' * 60)

    baseline = DenoisingAutoencoder().to(device)
    baseline_total_params = count_params(baseline)
    print(f'Baseline model params: {baseline_total_params:,}')

    psnr_base = {}

    for t in range(NUM_TASKS):
        print(f'\n{"─"*50}')
        print(f' Task {t}: digits {{{task_name(t)}}}')
        print(f'{"─"*50}')

        train_ld, _ = get_split_task_loaders(t, args.batch_size)

        opt = optim.Adam(baseline.parameters(), lr=args.lr)
        sched = optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs_per_task)

        for ep in range(1, args.epochs_per_task + 1):
            loss = train_epoch(baseline, train_ld, criterion, opt, device)
            sched.step()
            if ep % 10 == 0 or ep == args.epochs_per_task:
                print(f'    ep {ep:3d}  loss={loss:.6f}')

        psnr_base[t] = {}
        for j in range(NUM_TASKS):
            psnr_base[t][j] = compute_psnr(baseline, test_loaders[j], device)

        print(f'  PSNR after Task {t}:')
        for j in range(NUM_TASKS):
            tag = ' (unseen)' if j > t else ''
            print(f'    T{j} ({task_name(j):>3s}): '
                  f'{psnr_base[t][j]:.2f} dB{tag}')

    # ════════════════════════════════════════════════════════════════════════
    #  C. Memory Analysis
    # ════════════════════════════════════════════════════════════════════════
    atom_per_task = count_atom_scalars(task_atoms[0])
    shared_params = dcf_total_params - atom_per_task
    dcf_memory    = shared_params + NUM_TASKS * atom_per_task
    base_memory   = NUM_TASKS * baseline_total_params
    compression   = base_memory / dcf_memory

    print(f'\n{"=" * 60}')
    print('  Memory Analysis')
    print(f'{"=" * 60}')
    print(f'  DCF shared (coeffs + decoder + BN):  {shared_params:,}')
    print(f'  DCF atoms per task:                  {atom_per_task:,}')
    print(f'  DCF-CL total ({NUM_TASKS} tasks):            {dcf_memory:,}')
    print(f'  Baseline total ({NUM_TASKS} tasks):           {base_memory:,}')
    print(f'  Compression ratio:                   {compression:.2f}x')

    # ════════════════════════════════════════════════════════════════════════
    #  D. Backward Transfer
    # ════════════════════════════════════════════════════════════════════════
    T = NUM_TASKS

    # DCF-CL: exact restore ⟹ BWT = 0.0 by construction
    bwt_dcf = 0.0

    # Verify: compare peak PSNR to final PSNR for DCF-CL
    print(f'\nDCF-CL zero-forgetting verification:')
    for j in range(T):
        peak  = psnr_dcf[j][j]
        final = psnr_dcf[T - 1][j]
        diff  = final - peak
        print(f'  T{j}: peak={peak:.2f}  final={final:.2f}  '
              f'diff={diff:+.4f} dB')

    # Baseline BWT
    bwt_sum = 0.0
    for j in range(T - 1):
        peak_j  = psnr_base[j][j]
        final_j = psnr_base[T - 1][j]
        bwt_sum += (final_j - peak_j)
    bwt_base = bwt_sum / (T - 1)

    print(f'\nBackward Transfer:')
    print(f'  DCF-CL BWT:   {bwt_dcf:+.2f} dB')
    print(f'  Baseline BWT: {bwt_base:+.2f} dB')

    # ════════════════════════════════════════════════════════════════════════
    #  E. Plots
    # ════════════════════════════════════════════════════════════════════════
    print(f'\nGenerating plots...')
    plot_blur_examples(test_loaders, device)
    plot_forgetting_curves(psnr_dcf, psnr_base)
    plot_psnr_heatmaps(psnr_dcf, psnr_base)
    plot_memory_footprint(shared_params, atom_per_task, baseline_total_params)

    # ════════════════════════════════════════════════════════════════════════
    #  F. Save JSON
    # ════════════════════════════════════════════════════════════════════════
    results = {
        'args': vars(args),
        'dcf_total_params':      dcf_total_params,
        'baseline_total_params': baseline_total_params,
        'atom_params_per_task':  atom_per_task,
        'shared_params':         shared_params,
        'dcf_total_memory':      dcf_memory,
        'baseline_total_memory': base_memory,
        'compression_ratio':     compression,
        'bwt_dcf':               bwt_dcf,
        'bwt_baseline':          bwt_base,
        'psnr_dcf': {str(t): {str(j): round(v, 4)
                     for j, v in row.items()}
                     for t, row in psnr_dcf.items()},
        'psnr_baseline': {str(t): {str(j): round(v, 4)
                          for j, v in row.items()}
                          for t, row in psnr_base.items()},
        'final_psnr_dcf': {str(j): round(psnr_dcf[T-1][j], 4)
                           for j in range(T)},
        'final_psnr_baseline': {str(j): round(psnr_base[T-1][j], 4)
                                for j in range(T)},
    }

    out = os.path.join(RESULTS, 'cl_deblur_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {out}')
    print('Done.')


if __name__ == '__main__':
    main()
