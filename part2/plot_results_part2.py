"""
Generate all plots for Part 2.

Required input files (in results_part2/):
    baseline_results.json
    svhn_direct_results.json
    adaptation_results.json
    atoms_mnist_K{K}.npy          (for some representative K)
    atoms_svhn_expB_K{K}.npy

Output plots (saved to results_part2/):
    fig_p2_accuracy_vs_K.pdf          — SVHN test accuracy vs K (all methods)
    fig_p2_summary_bar.pdf            — bar chart: Task 1 / 2A / 2B / 2C (best K)
    fig_p2_atoms_comparison_K{K}.pdf  — MNIST vs SVHN atoms side-by-side
    fig_p2_pre_vs_adapted.pdf         — pre-adapt vs adapted SVHN accuracy vs K

Usage:
    python plot_results_part2.py [--rep_K 4]
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_json(fname):
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        print(f'[WARN] {path} not found — skipping dependent plots.')
        return None
    with open(path) as f:
        return json.load(f)


def load_npy(fname):
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        print(f'[WARN] {path} not found.')
        return None
    return np.load(path)


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
        'lines.linewidth': 2,
        'lines.markersize': 7,
    })


# ── Fig 1 — SVHN Accuracy vs K (all DCF experiments) ─────────────────────────

def plot_accuracy_vs_K(direct, adapt):
    fig, ax = plt.subplots(figsize=(9, 5))

    K_ticks = []
    if direct:
        K  = direct['results']['K_list']
        a  = direct['results']['best_test_acc']
        ax.plot(K, a, 's-', color='tab:red',   label='Exp A: DCF trained on SVHN (upper bound)')
        K_ticks = K

    if adapt:
        K      = adapt['results']['K_list']
        a_pre  = adapt['results']['pre_adapt_svhn_acc']
        a_expB = adapt['results']['expB_svhn_acc']
        a_expC = adapt['results']['expC_svhn_acc']
        ax.plot(K, a_expB, 'o-', color='tab:blue',
                label='Exp B: Adapt atoms $\\psi_k$  (freeze $a_k$)')
        ax.plot(K, a_expC, '^--', color='tab:orange',
                label='Exp C: Adapt coefficients $a_k$  (freeze $\\psi_k$)')
        ax.plot(K, a_pre,  'D:', color='gray', alpha=0.7,
                label='MNIST$\\to$SVHN (no adaptation)')
        K_ticks = K

    ax.set_xlabel('Number of Filter Atoms $K$')
    ax.set_ylabel('SVHN Test Accuracy (%)')
    ax.set_title('Part 2: SVHN Test Accuracy vs $K$\n(DCF Domain Adaptation Experiments)')
    ax.set_xticks([int(k) for k in K_ticks])
    ax.legend(fontsize=9, loc='center right')
    ax.grid(True, alpha=0.3)
    savefig(fig, 'fig_p2_accuracy_vs_K.pdf')


# ── Fig 2 — Summary bar chart ─────────────────────────────────────────────────

def plot_summary_bar(baseline, direct, adapt, rep_K=8):
    """Bar chart comparing all four conditions at a representative K."""
    if not all([baseline, direct, adapt]):
        print('[WARN] Some results missing — skipping summary bar chart.')
        return

    # Find index of rep_K in the results lists
    def _get_acc(results_dict, key, K):
        try:
            idx = results_dict['results']['K_list'].index(K)
            return results_dict['results'][key][idx]
        except (ValueError, IndexError):
            return results_dict['results'][key][-1]   # fall back to last K

    no_adapt = baseline.get('svhn_no_adapt_acc', 0.0)
    upper    = _get_acc(direct, 'best_test_acc', rep_K)
    expB     = _get_acc(adapt,  'expB_svhn_acc',  rep_K)
    expC     = _get_acc(adapt,  'expC_svhn_acc',  rep_K)

    labels = [
        f'Task 1\n(Baseline,\nno adapt)',
        f'Task 2A\n(Train on SVHN,\nK={rep_K})',
        f'Task 2B\n(Adapt atoms,\nK={rep_K})',
        f'Task 2C\n(Adapt coeff.,\nK={rep_K})',
    ]
    accs   = [no_adapt, upper, expB, expC]
    colors = ['tab:gray', 'tab:red', 'tab:blue', 'tab:orange']

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, accs, color=colors, width=0.55, edgecolor='white', linewidth=1.2)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('SVHN Test Accuracy (%)')
    ax.set_title(f'Part 2: Summary of SVHN Accuracy (representative K={rep_K})')
    ax.set_ylim(0, max(accs) * 1.12)
    ax.grid(axis='y', alpha=0.3)
    savefig(fig, 'fig_p2_summary_bar.pdf')


# ── Fig 3 — Atom visualisation: MNIST vs SVHN-adapted ────────────────────────

def plot_atoms_comparison(K):
    atoms_mnist = load_npy(f'atoms_mnist_K{K}.npy')
    atoms_svhn  = load_npy(f'atoms_svhn_expB_K{K}.npy')
    if atoms_mnist is None or atoms_svhn is None:
        print(f'[WARN] Atom arrays for K={K} not found — skipping atom visualisation.')
        return

    n_atoms = atoms_mnist.shape[0]  # K
    fig, axes = plt.subplots(2, n_atoms, figsize=(n_atoms * 1.5, 3.2))

    vmax = max(np.abs(atoms_mnist).max(), np.abs(atoms_svhn).max())

    for k in range(n_atoms):
        # MNIST atoms (top row)
        axes[0, k].imshow(atoms_mnist[k], cmap='bwr', vmin=-vmax, vmax=vmax)
        axes[0, k].set_title(f'$\\psi_{{{k+1}}}$', fontsize=9)
        axes[0, k].axis('off')
        # SVHN-adapted atoms (bottom row, Exp B)
        axes[1, k].imshow(atoms_svhn[k], cmap='bwr', vmin=-vmax, vmax=vmax)
        axes[1, k].axis('off')

    # Row labels
    axes[0, 0].set_ylabel('MNIST', fontsize=10, labelpad=4)
    axes[1, 0].set_ylabel('SVHN\n(Exp B)', fontsize=10, labelpad=4)
    for ax in axes[:, 0]:
        ax.axis('on')
        ax.set_yticks([])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(f'Domain-Specific Filter Atoms (Conv1, K={K})\n'
                 f'MNIST atoms vs SVHN-adapted atoms (Exp B: adapt ψ_k, freeze a_k)',
                 y=1.04, fontsize=11)
    plt.tight_layout()
    savefig(fig, f'fig_p2_atoms_comparison_K{K}.pdf')


# ── Fig 4 — Pre-adaptation vs adapted SVHN accuracy ──────────────────────────

def plot_pre_vs_adapted(adapt):
    if adapt is None:
        return
    K      = adapt['results']['K_list']
    a_pre  = adapt['results']['pre_adapt_svhn_acc']
    a_expB = adapt['results']['expB_svhn_acc']
    a_expC = adapt['results']['expC_svhn_acc']

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(K, a_pre,  'D:', color='gray',       label='Pre-adaptation (no fine-tune)')
    ax.plot(K, a_expB, 'o-', color='tab:blue',   label='Exp B (adapt atoms)')
    ax.plot(K, a_expC, '^--', color='tab:orange', label='Exp C (adapt coefficients)')

    ax.fill_between(K, a_pre, a_expB, alpha=0.12, color='tab:blue',
                    label='Gain from Exp B adaptation')

    ax.set_xlabel('Number of Filter Atoms $K$')
    ax.set_ylabel('SVHN Test Accuracy (%)')
    ax.set_title('Pre-Adaptation vs Post-Adaptation SVHN Accuracy')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig(fig, 'fig_p2_pre_vs_adapted.pdf')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rep_K', type=int, default=8,
                   help='Representative K for summary bar chart')
    args = p.parse_args()

    style()
    baseline = load_json('baseline_results.json')
    direct   = load_json('svhn_direct_results.json')
    adapt    = load_json('adaptation_results.json')

    plot_accuracy_vs_K(direct, adapt)
    plot_summary_bar(baseline, direct, adapt, rep_K=args.rep_K)

    # Atom comparison for the representative K
    plot_atoms_comparison(args.rep_K)

    # Also plot for K=4 if different from rep_K
    if args.rep_K != 4 and adapt is not None and 4 in adapt['results']['K_list']:
        plot_atoms_comparison(4)

    plot_pre_vs_adapted(adapt)

    print('\nAll Part-2 plots saved to:', RESULTS_DIR)


if __name__ == '__main__':
    main()
