"""
Generate all plots for Part 1.

Required input files (in results/):
    pca_results.json
    dcf_results.json
    fb_results.json
    fb_atoms_K6.npy
    autoencoder_dcf_results.json
    autoencoder_fb_results.json

Output plots (saved to results/):
    fig1_accuracy_vs_K.pdf        — Task 1: test accuracy vs K for PCA/DCF
    fig2_params_vs_K.pdf          — Task 1: # parameters vs K
    fig3_fb_dcf_accuracy_vs_K.pdf — Task 2: accuracy comparison (FB vs DCF)
    fig4_fb_atoms.pdf             — Task 2: visualise 8 FB atoms
    fig5_dcf_atoms.pdf            — Task 2: visualise top-8 PCA filter atoms (layer 0)
    fig6_psnr_vs_K.pdf            — Task 3: PSNR vs K for DCF and FB autoencoders

Usage:
    python plot_results.py
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


# ── Helpers ──────────────────────────────────────────────────────────────────

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
        'lines.linewidth': 2,
        'lines.markersize': 7,
    })


# ── Fig 1 — Test Accuracy vs K (Task 1) ─────────────────────────────────────

def plot_accuracy_vs_K(pca, dcf):
    fig, ax = plt.subplots(figsize=(6, 4))

    if pca:
        K_pca = pca['K_list']
        acc_pca = pca['test_acc']
        baseline = pca.get('baseline_test_acc', None)
        ax.plot(K_pca, acc_pca, 'o-', label='PCA (post-hoc, no retraining)', color='tab:blue')
        if baseline:
            ax.axhline(baseline, linestyle='--', color='tab:blue', alpha=0.5,
                       label=f'Baseline (full weights, {baseline:.2f}%)')

    if dcf:
        K_dcf = dcf['results']['K_list']
        acc_dcf = dcf['results']['best_test_acc']
        ax.plot(K_dcf, acc_dcf, 's-', label='DCF (learned atoms, trained)', color='tab:orange')

    ax.set_xlabel('Number of Filter Atoms K')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('MNIST Test Accuracy vs Number of Filter Atoms K\n(AlexNet, 3×3 kernels)')
    ax.legend(fontsize=9)
    ax.set_xticks(range(1, 10))
    ax.grid(True, alpha=0.3)
    savefig(fig, 'fig1_accuracy_vs_K.pdf')


# ── Fig 2 — Parameters vs K (Task 1) ─────────────────────────────────────────

def plot_params_vs_K(pca, dcf):
    fig, ax = plt.subplots(figsize=(6, 4))

    if pca:
        K_pca    = pca['K_list']
        p_pca    = pca['num_conv_params']
        baseline = pca.get('baseline_conv_params', None)
        ax.plot(K_pca, p_pca, 'o-', label='DCF/PCA conv params', color='tab:blue')
        if baseline:
            ax.axhline(baseline, linestyle='--', color='gray',
                       label=f'Baseline conv params ({baseline:,})')

    if dcf:
        K_dcf  = dcf['results']['K_list']
        p_dcf  = dcf['results']['conv_params']
        ax.plot(K_dcf, p_dcf, 's-', label='DCF (learned) conv params', color='tab:orange')

    ax.set_xlabel('Number of Filter Atoms K')
    ax.set_ylabel('Number of Convolutional Parameters')
    ax.set_title('Conv Layer Parameter Count vs K')
    ax.legend(fontsize=9)
    ax.set_xticks(range(1, 10))
    ax.yaxis.get_major_formatter().set_scientific(False)
    ax.grid(True, alpha=0.3)
    savefig(fig, 'fig2_params_vs_K.pdf')


# ── Fig 3 — Accuracy: FB vs DCF (Task 2) ─────────────────────────────────────

def plot_fb_dcf_accuracy(pca, dcf, fb):
    fig, ax = plt.subplots(figsize=(9, 5))

    all_K = []
    if fb:
        K_fb  = fb['results']['K_list']
        a_fb  = fb['results']['best_test_acc']
        ax.plot(K_fb, a_fb, '^-', label='FB (fixed atoms, learned coefficients)',
                color='tab:green')
        all_K.extend(K_fb)

    if dcf:
        K_dcf = dcf['results']['K_list']
        a_dcf = dcf['results']['best_test_acc']
        ax.plot(K_dcf, a_dcf, 's-', label='DCF (learned atoms + coefficients)',
                color='tab:orange')
        all_K.extend(K_dcf)

    if pca:
        K_pca = pca['K_list']
        a_pca = pca['test_acc']
        ax.plot(K_pca, a_pca, 'o--', label='PCA (post-hoc reconstruction)',
                color='tab:blue', alpha=0.7)
        all_K.extend(K_pca)
        if pca.get('baseline_test_acc'):
            ax.axhline(pca['baseline_test_acc'], linestyle=':', color='gray',
                       label=f"Baseline ({pca['baseline_test_acc']:.2f}%)")

    ax.set_xlabel('Number of Components K')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Task 2: FB Bases vs DCF (Learned Atoms)\nMNIST Classification Accuracy')
    ax.set_xticks(sorted(set(int(k) for k in all_K)))
    ax.legend(fontsize=9, bbox_to_anchor=(1.05, 0.5), loc='center left',
              borderaxespad=0)
    ax.grid(True, alpha=0.3)
    savefig(fig, 'fig3_fb_dcf_accuracy_vs_K.pdf')


# ── Fig 4 — Visualise FB Atoms ────────────────────────────────────────────────

def plot_fb_atoms():
    # Accept either fb_atoms_K6.npy (current) or fb_atoms_K8.npy (legacy)
    path = None
    for candidate in ['fb_atoms_K6.npy', 'fb_atoms_K8.npy']:
        p = os.path.join(RESULTS_DIR, candidate)
        if os.path.exists(p):
            path = p
            break
    if path is None:
        print(f'[WARN] fb_atoms_K*.npy not found in {RESULTS_DIR} — skipping Fig 4.')
        return
    atoms = np.load(path)
    K = atoms.shape[0]
    fig, axes = plt.subplots(1, K, figsize=(K * 1.4, 1.8))
    for k, ax in enumerate(axes):
        im = ax.imshow(atoms[k], cmap='bwr',
                       vmin=-np.abs(atoms).max(), vmax=np.abs(atoms).max())
        ax.set_title(f'$\\psi_{{{k+1}}}$', fontsize=9)
        ax.axis('off')
    fig.suptitle(f'Fourier-Bessel Bases (3×3, K={K})', y=1.02)
    plt.tight_layout()
    savefig(fig, 'fig4_fb_atoms.pdf')


# ── Fig 5 — Visualise PCA / DCF Atoms ────────────────────────────────────────

def plot_pca_atoms(pca):
    if pca is None:
        return
    atoms_data = pca.get('atoms', {}).get('features_conv0_atoms', None)
    if atoms_data is None:
        print('[WARN] PCA atoms not found in pca_results.json — skipping Fig 5.')
        return
    atoms = np.array(atoms_data)   # (8, 3, 3)
    K = atoms.shape[0]
    fig, axes = plt.subplots(1, K, figsize=(K * 1.4, 1.8))
    for k, ax in enumerate(axes):
        vmax = np.abs(atoms[k]).max() or 1e-6
        ax.imshow(atoms[k], cmap='bwr', vmin=-vmax, vmax=vmax)
        ax.set_title(f'$\\psi_{{{k+1}}}$', fontsize=9)
        ax.axis('off')
    fig.suptitle('PCA Filter Atoms — AlexNet Conv1 (top-8 principal components)', y=1.02)
    plt.tight_layout()
    savefig(fig, 'fig5_dcf_atoms.pdf')


# ── Fig 6 — PSNR vs K (Task 3) ───────────────────────────────────────────────

def plot_psnr_vs_K(dcf_ae, fb_ae):
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False

    if dcf_ae:
        K_dcf = dcf_ae['results']['K_list']
        p_dcf = dcf_ae['results']['best_psnr']
        ax.plot(K_dcf, p_dcf, 's-', label='DCF autoencoder (learned atoms)',
                color='tab:orange')
        plotted = True

    if fb_ae:
        K_fb = fb_ae['results']['K_list']
        p_fb = fb_ae['results']['best_psnr']
        ax.plot(K_fb, p_fb, '^-', label='FB autoencoder (fixed atoms)',
                color='tab:green')
        plotted = True

    if not plotted:
        print('[WARN] No autoencoder results found — skipping Fig 6.')
        plt.close(fig)
        return

    ax.set_xlabel('Number of Components K')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Task 3: PSNR vs K — Denoising Autoencoder on MNIST')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig(fig, 'fig6_psnr_vs_K.pdf')


# ── Fig 7 — Explained Variance (bonus, helps with Discussion) ────────────────

def plot_explained_variance(pca):
    if pca is None:
        return
    evr = pca.get('layer_explained_variance', {})
    if not evr:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, vals in evr.items():
        K_range = list(range(1, len(vals) + 1))
        short   = name.replace('features.', 'conv')
        ax.plot(K_range, [v * 100 for v in vals], marker='o', label=short)
    ax.set_xlabel('Number of Components K')
    ax.set_ylabel('Cumulative Explained Variance (%)')
    ax.set_title('PCA Explained Variance by Conv Layer')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig(fig, 'fig7_explained_variance.pdf')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    style()
    pca    = load_json('pca_results.json')
    dcf    = load_json('dcf_results.json')
    fb     = load_json('fb_results.json')
    dcf_ae = load_json('autoencoder_dcf_results.json')
    fb_ae  = load_json('autoencoder_fb_results.json')

    plot_accuracy_vs_K(pca, dcf)
    plot_params_vs_K(pca, dcf)
    plot_fb_dcf_accuracy(pca, dcf, fb)
    plot_fb_atoms()
    plot_pca_atoms(pca)
    plot_psnr_vs_K(dcf_ae, fb_ae)
    plot_explained_variance(pca)

    print('\nAll plots generated in:', RESULTS_DIR)


if __name__ == '__main__':
    main()
