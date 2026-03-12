"""
Part 3 — DCF Continual-Learning Deblurring via Coefficient Swapping.

Procedure (following the atom-swapping idea of Miao et al., ICLR 2022):

  Task 0 (initial):
      Build DCF autoencoder with K atoms, bases_grad=True.
      Train BOTH atoms ψ_k and coefficients a_k^{(0)} on Task-0 blur.
      Save a_k^{(0)} (+ BN + decoder state) as "Task-0 coefficients".
      FREEZE ψ_k for all future tasks.

  Task t > 0:
      Re-initialise task-specific parameters (a_k, BN, decoder).
      Unfreeze them; keep ψ_k frozen.
      Train on Task-t blur for E epochs.
      Save a_k^{(t)} as "Task-t coefficients".

  Evaluation for task j:
      Inject a_k^{(j)} into the model → evaluate PSNR on Task-j test set.

Because ψ_k never changes after Task 0, swapping coefficients is an
exact restore — there is ZERO forgetting by construction.

Outputs (saved to results_part3/):
    dcf_cl_results.json        — full PSNR matrix + memory analysis
    dcf_shared_atoms.pth       — shared atom state after Task 0
    dcf_task{0,1,2}_coeff.pth  — task-specific coefficient dicts

Usage:
    python train_cl_dcf.py [--num_bases 6] [--epochs_per_task 20] [--lr 1e-3]
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
from models.autoencoder_dcf import DenoisingAutoencoderDCF
from cl_utils import (
    extract_task_coefficients, inject_task_coefficients,
    count_coefficients_memory, count_all_params, count_shared_atoms,
    freeze_atoms, unfreeze_task_specific, unfreeze_all, compute_psnr,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results_part3')
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--num_bases',       type=int,   default=6)
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


def reinit_task_specific(model):
    """
    Re-initialise all task-specific parameters (coefficients, BN, decoder)
    so each new task starts from a fresh state instead of the previous
    task's parameters.
    """
    from dcf_layer import Conv_DCF
    for m in model.modules():
        if isinstance(m, Conv_DCF):
            nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available()
                          else 'cpu')
    print(f'Device: {device}')
    print(f'K = {args.num_bases}\n')

    model = DenoisingAutoencoderDCF(
        num_bases=args.num_bases, bases_grad=True, initializer='random'
    ).to(device)

    criterion = nn.MSELoss()

    total_params = count_all_params(model)
    atom_params  = count_shared_atoms(model)
    print(f'Total model params:  {total_params:,}')
    print(f'Shared atom params:  {atom_params:,}')

    # Pre-load test loaders for all tasks
    test_loaders = {}
    for t in range(NUM_TASKS):
        _, tl = get_blur_task_loaders(t, args.batch_size)
        test_loaders[t] = tl

    # Storage for task-specific coefficient snapshots
    task_coefficients = {}
    # PSNR matrix: psnr_after[t][j] = PSNR on task j after training task t
    psnr_after = {}

    for t in range(NUM_TASKS):
        print(f'\n{"="*55}')
        print(f' Task {t}: {task_name(t)}')
        print(f'{"="*55}')

        train_ld, _ = get_blur_task_loaders(t, args.batch_size)

        if t == 0:
            # ── Task 0: learn both atoms and coefficients ────────────────────
            unfreeze_all(model)
            trainable = sum(p.numel() for p in model.parameters()
                            if p.requires_grad)
            print(f'  Task 0: training ALL params ({trainable:,})')
        else:
            # ── Task t > 0: freeze atoms, re-init & unfreeze task-specific ──
            reinit_task_specific(model)
            freeze_atoms(model)
            unfreeze_task_specific(model)
            trainable = sum(p.numel() for p in model.parameters()
                            if p.requires_grad)
            print(f'  Task {t}: training task-specific params ({trainable:,}), '
                  f'atoms FROZEN')

        params_to_opt = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params_to_opt, lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs_per_task)

        for ep in range(1, args.epochs_per_task + 1):
            loss = train_one_epoch(model, train_ld, criterion,
                                   optimizer, device)
            scheduler.step()
            if ep % 5 == 0 or ep == args.epochs_per_task:
                print(f'    ep {ep:3d}  loss={loss:.6f}')

        # Save atoms after Task 0
        if t == 0:
            atom_state = {}
            from dcf_layer import Conv_DCF
            for name, m in model.named_modules():
                if isinstance(m, Conv_DCF):
                    atom_state[name + '.bases'] = m.bases.detach().cpu().clone()
            torch.save(atom_state,
                       os.path.join(RESULTS_DIR, 'dcf_shared_atoms.pth'))

        # Snapshot task-specific coefficients
        coeff = extract_task_coefficients(model)
        task_coefficients[t] = coeff
        torch.save(coeff,
                   os.path.join(RESULTS_DIR, f'dcf_task{t}_coeff.pth'))
        coeff_mem = count_coefficients_memory(coeff)
        print(f'  Task {t} coefficient memory: {coeff_mem:,} scalars')

        # ── Evaluate on ALL tasks by swapping coefficients ───────────────────
        psnr_row = {}
        for j in range(NUM_TASKS):
            inject_task_coefficients(model, task_coefficients[min(j, t)])
            p = compute_psnr(model, test_loaders[j], device)
            psnr_row[j] = p
        psnr_after[t] = psnr_row

        # Restore current task state
        inject_task_coefficients(model, task_coefficients[t])

        print(f'\n  PSNR after Task {t} (via coefficient swapping):')
        for j in range(NUM_TASKS):
            src = min(j, t)
            tag = f' [swap in Task-{src} coeff]' if j != t else ''
            tag2 = ' (not yet seen)' if j > t else ''
            print(f'    Task {j} ({task_name(j):>8s}): '
                  f'{psnr_row[j]:.2f} dB{tag}{tag2}')

    # ── Final evaluation with explicit coefficient swapping ──────────────────
    print(f'\n{"="*55}')
    print('Final DCF-CL evaluation (all tasks, swapping coefficients):')
    print(f'{"="*55}')
    final_psnr = {}
    for j in range(NUM_TASKS):
        inject_task_coefficients(model, task_coefficients[j])
        p = compute_psnr(model, test_loaders[j], device)
        final_psnr[j] = p
        print(f'  Task {j} ({task_name(j):>8s}): {p:.2f} dB')

    # ── Memory footprint analysis ────────────────────────────────────────────
    coeff_per_task = {t: count_coefficients_memory(task_coefficients[t])
                      for t in range(NUM_TASKS)}
    total_dcf_memory = atom_params + sum(coeff_per_task.values())
    naive_memory = total_params * NUM_TASKS

    print(f'\nMemory footprint:')
    print(f'  Shared atoms:              {atom_params:,}')
    for t in range(NUM_TASKS):
        print(f'  Task-{t} coefficients:       {coeff_per_task[t]:,}')
    print(f'  DCF-CL total:              {total_dcf_memory:,}')
    print(f'  Naive (separate models):   {naive_memory:,}')
    print(f'  Compression ratio:         {naive_memory / total_dcf_memory:.2f}×')

    # ── Save everything ──────────────────────────────────────────────────────
    results = {
        'args': vars(args),
        'total_model_params': total_params,
        'shared_atom_params': atom_params,
        'coeff_per_task': {str(t): v for t, v in coeff_per_task.items()},
        'total_dcf_memory': total_dcf_memory,
        'naive_memory': naive_memory,
        'compression_ratio': naive_memory / total_dcf_memory,
        'psnr_after': {str(t): {str(j): v for j, v in row.items()}
                       for t, row in psnr_after.items()},
        'final_psnr': {str(j): v for j, v in final_psnr.items()},
    }
    out = os.path.join(RESULTS_DIR, 'dcf_cl_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {out}')


if __name__ == '__main__':
    main()
