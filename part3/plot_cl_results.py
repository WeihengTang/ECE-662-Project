"""
Generate all plots for Part 3 — Continual Deblurring.

Required input files (in results_part3/):
    baseline_cl_results.json
    dcf_cl_results.json

Output plots (saved to results_part3/):
    fig_p3_forgetting_curves.pdf    — PSNR vs task sequence for baseline & DCF
    fig_p3_memory_footprint.pdf     — bar chart: memory per task
    fig_p3_psnr_heatmaps.pdf        — two heatmaps (baseline vs DCF)
    fig_p3_blur_examples.pdf         — example blurred / reconstructed images

Usage:
    python plot_cl_results.py
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_blur import TASK_DEFS, NUM_TASKS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'results_part3')
TASK_NAMES = [d[0] for d in TASK_DEFS]


def load_json(fname):
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        print(f'[WARN] {path} not found — skipping dependent plots.')
        return None
    with open(path) as f:
        return json.load(f)


def savefig(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    print(f'Saved: {path}')
    plt.close(fig)


def style():
    plt.rcParams.update({
        'font.size': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 2.2,
        'lines.markersize': 8,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Forgetting curves: PSNR on each task after each training phase
# ═══════════════════════════════════════════════════════════════════════════════

def plot_forgetting_curves(bl, dcf):
    if bl is None and dcf is None:
        return
    fig, axes = plt.subplots(1, NUM_TASKS, figsize=(5 * NUM_TASKS, 4.2),
                             sharey=True)

    for j in range(NUM_TASKS):
        ax = axes[j]
        xs = list(range(NUM_TASKS))

        if bl is not None:
            psnr_bl = []
            for t in range(NUM_TASKS):
                val = bl['psnr_after'].get(str(t), {}).get(str(j), None)
                psnr_bl.append(val)
            # Only plot from the task j was first seen
            ys = [v if v is not None else np.nan for v in psnr_bl]
            ax.plot(xs, ys, 'o--', color='tab:red', label='Baseline (naive)')

        if dcf is not None:
            psnr_dcf = []
            for t in range(NUM_TASKS):
                val = dcf['psnr_after'].get(str(t), {}).get(str(j), None)
                psnr_dcf.append(val)
            ys = [v if v is not None else np.nan for v in psnr_dcf]
            ax.plot(xs, ys, 's-', color='tab:blue', label='DCF-CL (ours)')

        ax.set_xlabel('Training phase (after Task $t$)')
        ax.set_xticks(xs)
        ax.set_xticklabels([f'T{t}' for t in xs])
        ax.set_title(f'PSNR on Task {j}\n({TASK_NAMES[j]})', fontsize=11)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.set_ylabel('PSNR (dB)')
        if j == NUM_TASKS - 1:
            ax.legend(fontsize=9, loc='lower left')

    fig.suptitle('Forgetting Curves: PSNR on Each Task After Sequential Training',
                 y=1.02, fontsize=13, fontweight='bold')
    plt.tight_layout()
    savefig(fig, 'fig_p3_forgetting_curves.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Memory footprint comparison
# ═══════════════════════════════════════════════════════════════════════════════

def plot_memory_footprint(bl, dcf):
    if bl is None and dcf is None:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Baseline: stores a full model per task (naive approach)
    bl_per_task = bl['params_per_task'] if bl else 0
    bl_total = bl_per_task * NUM_TASKS if bl else 0

    # DCF-CL: shared atoms + per-task coefficients
    dcf_atoms = dcf['shared_atom_params'] if dcf else 0
    dcf_coeff_per_task = [dcf['coeff_per_task'].get(str(t), 0)
                          for t in range(NUM_TASKS)] if dcf else [0] * NUM_TASKS
    dcf_total = dcf['total_dcf_memory'] if dcf else 0

    x = np.arange(2)
    width = 0.5

    # Stacked bars: baseline = full model × 3; DCF = atoms + coeff×3
    if bl is not None:
        bars_bl = [bl_per_task] * NUM_TASKS
        bottom = 0
        for t in range(NUM_TASKS):
            ax.bar(0, bars_bl[t], width, bottom=bottom,
                   color=f'C{t}', alpha=0.7,
                   label=f'Task {t} ({TASK_NAMES[t]})' if t == 0 else
                         f'Task {t} ({TASK_NAMES[t]})')
            bottom += bars_bl[t]

    if dcf is not None:
        # Shared atoms at the bottom
        ax.bar(1, dcf_atoms, width, color='tab:gray', alpha=0.8,
               label='Shared atoms ψ_k')
        bottom = dcf_atoms
        for t in range(NUM_TASKS):
            ax.bar(1, dcf_coeff_per_task[t], width, bottom=bottom,
                   color=f'C{t}', alpha=0.7)
            bottom += dcf_coeff_per_task[t]

    ax.set_xticks(x)
    ax.set_xticklabels(['Baseline\n(separate models)', 'DCF-CL\n(ours)'],
                       fontsize=11)
    ax.set_ylabel('Total Stored Parameters')
    ax.yaxis.get_major_formatter().set_scientific(False)

    # Annotate totals
    if bl is not None:
        ax.text(0, bl_total * 1.02, f'{bl_total:,}',
                ha='center', fontsize=10, fontweight='bold')
    if dcf is not None:
        ax.text(1, dcf_total * 1.02, f'{dcf_total:,}',
                ha='center', fontsize=10, fontweight='bold')
        ratio = bl_total / dcf_total if dcf_total > 0 else 0
        ax.text(1, dcf_total * 1.08,
                f'({ratio:.1f}× compression)',
                ha='center', fontsize=9, color='tab:blue')

    ax.set_title('Memory Footprint: Baseline vs DCF-CL',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    savefig(fig, 'fig_p3_memory_footprint.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 — PSNR heatmaps (baseline vs DCF)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_psnr_heatmaps(bl, dcf):
    if bl is None and dcf is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for idx, (data, label) in enumerate([(bl, 'Baseline'), (dcf, 'DCF-CL')]):
        ax = axes[idx]
        if data is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(label)
            continue

        mat = np.zeros((NUM_TASKS, NUM_TASKS))
        for t in range(NUM_TASKS):
            for j in range(NUM_TASKS):
                val = data['psnr_after'].get(str(t), {}).get(str(j), np.nan)
                mat[t, j] = val

        im = ax.imshow(mat, cmap='YlOrRd', aspect='auto',
                       vmin=np.nanmin(mat) - 1, vmax=np.nanmax(mat) + 1)
        # Annotate cells
        for t in range(NUM_TASKS):
            for j in range(NUM_TASKS):
                v = mat[t, j]
                if not np.isnan(v):
                    ax.text(j, t, f'{v:.1f}', ha='center', va='center',
                            fontsize=10, fontweight='bold')

        ax.set_xticks(range(NUM_TASKS))
        ax.set_xticklabels([f'Task {j}\n{TASK_NAMES[j]}' for j in range(NUM_TASKS)],
                           fontsize=9)
        ax.set_yticks(range(NUM_TASKS))
        ax.set_yticklabels([f'After T{t}' for t in range(NUM_TASKS)],
                           fontsize=9)
        ax.set_xlabel('Evaluated on')
        ax.set_ylabel('Trained through')
        ax.set_title(label, fontsize=12, fontweight='bold')
        fig.colorbar(im, ax=ax, label='PSNR (dB)', shrink=0.8)

    fig.suptitle('PSNR Matrix: Row = Training Phase, Column = Evaluation Task',
                 y=1.03, fontsize=13)
    plt.tight_layout()
    savefig(fig, 'fig_p3_psnr_heatmaps.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Example blur kernels and degraded images  (no model needed)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_blur_examples():
    """Visualise the three blur kernels and example degraded digits."""
    from dataset_blur import get_blur_kernel, BlurredMNIST

    fig, axes = plt.subplots(2, NUM_TASKS, figsize=(4 * NUM_TASKS, 6))

    for t in range(NUM_TASKS):
        # Kernel
        kernel = get_blur_kernel(t).squeeze().numpy()
        axes[0, t].imshow(kernel, cmap='hot', interpolation='nearest')
        axes[0, t].set_title(f'Task {t}: {TASK_NAMES[t]}\nkernel',
                             fontsize=10)
        axes[0, t].axis('off')

        # Example blurred image
        ds = BlurredMNIST(t, train=False)
        blurred, clean = ds[0]
        axes[1, t].imshow(blurred.squeeze().numpy(), cmap='gray',
                          vmin=0, vmax=1)
        axes[1, t].set_title('blurred digit', fontsize=10)
        axes[1, t].axis('off')

    fig.suptitle('Synthetic Blur Tasks (PSF kernels and example degraded digits)',
                 y=1.0, fontsize=13, fontweight='bold')
    plt.tight_layout()
    savefig(fig, 'fig_p3_blur_examples.pdf')


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    style()
    bl  = load_json('baseline_cl_results.json')
    dcf = load_json('dcf_cl_results.json')

    plot_forgetting_curves(bl, dcf)
    plot_memory_footprint(bl, dcf)
    plot_psnr_heatmaps(bl, dcf)

    # Blur examples can always be generated (no trained model needed)
    try:
        plot_blur_examples()
    except Exception as e:
        print(f'[WARN] Could not generate blur examples: {e}')

    print(f'\nAll Part-3 plots saved to: {RESULTS_DIR}')


if __name__ == '__main__':
    main()
