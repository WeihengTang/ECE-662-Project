# Part 2 — Execution Instructions (GPU Cluster)

> Run all commands from the `part2/` directory unless otherwise noted.
> Part 2 imports models and DCF layers from `../part1/`, so `part1/` must be present
> in the same parent directory.

---

## 0. Environment Setup

Same environment as Part 1 (if already set up, skip this):

```bash
conda activate ece662    # or re-create if needed:
# conda create -n ece662 python=3.9 -y && conda activate ece662
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install scipy numpy matplotlib

cd /path/to/ECE-662-Project/part2

# Quick sanity check (CPU-only, no training):
python -c "from datasets_part2 import get_mnist_loaders; print('datasets OK')"
python -c "from adapt_utils import freeze_coefficients_unfreeze_atoms; print('adapt_utils OK')"
```

---

## 1. Task 1 — Baseline (Train on MNIST, Test on SVHN)

```bash
python train_baseline_part2.py \
    --epochs 30 --lr 0.01 --batch_size 128 --gpu 0 --seed 42
```

What this does:
1. Downloads MNIST and SVHN automatically to `./data/`.
2. Trains AlexNetMNIST for 30 epochs on MNIST.
3. Evaluates the best MNIST model on SVHN (no adaptation).
4. Prints the domain-shift accuracy drop.

Expected outputs:
- `results_part2/baseline_mnist_best.pth`
- `results_part2/baseline_results.json`

Expected result: SVHN accuracy ~20–40% (poor, illustrates domain shift).

---

## 2. Task 2A — DCF-AlexNet Trained Directly on SVHN (Upper Bound)

```bash
python train_dcf_svhn_direct.py \
    --epochs 30 --lr 0.01 --batch_size 128 --gpu 0 --seed 42
```

What this does:
- Trains DCF-AlexNet on SVHN from scratch for each K ∈ {1, 2, 4, 6, 8}.
- Reports best SVHN test accuracy per K.

Expected outputs:
- `results_part2/svhn_direct_results.json`

Expected SVHN accuracy: ~80–88% for K=8 (upper bound reference).

> Wall-clock estimate: ~5 K-values × 30 epochs × ~120s/epoch ≈ 5 hours.

---

## 3. Task 2B & 2C — Two-Stage DCF Adaptation

```bash
python train_dcf_adaptation.py \
    --epochs_s1 30 --epochs_s2 15 \
    --lr_s1 0.01  --lr_s2 0.001  \
    --batch_size 128 --gpu 0 --seed 42
```

What this does:
- **Stage 1 (shared):** Trains DCF-AlexNet on MNIST for each K.
  Saves checkpoint `results_part2/dcf_mnist_K{K}.pth`.
  Saves MNIST Conv1 atoms `results_part2/atoms_mnist_K{K}.npy`.
- **Stage 2B (Exp B):** Loads Stage-1 model, freezes `a_k` (coefficients),
  fine-tunes `ψ_k` (atoms) on SVHN for 15 epochs.
  Saves adapted model + SVHN Conv1 atoms.
- **Stage 2C (Exp C, ablation):** Loads Stage-1 model, freezes `ψ_k` (atoms),
  fine-tunes `a_k` (coefficients) on SVHN for 15 epochs.

Expected outputs:
- `results_part2/adaptation_results.json`
- `results_part2/dcf_mnist_K{1,2,4,6,8}.pth`
- `results_part2/dcf_svhn_expB_K{1,2,4,6,8}.pth`
- `results_part2/dcf_svhn_expC_K{1,2,4,6,8}.pth`
- `results_part2/atoms_mnist_K{K}.npy`         (for K ∈ {1,2,4,6,8})
- `results_part2/atoms_svhn_expB_K{K}.npy`

Expected SVHN accuracy:
- Exp B (adapt atoms):       ~55–70% (significant gain over no-adapt)
- Exp C (adapt coefficients): ~30–50% (worse than B; confirms theory)

> Wall-clock estimate: ~5K × (30+15) epochs × ~120s/epoch ≈ 7–8 hours.

---

## 4. Generate All Plots

Run **after** all training scripts have finished:

```bash
python plot_results_part2.py --rep_K 8
```

Expected outputs in `results_part2/`:
- `fig_p2_accuracy_vs_K.pdf`         — SVHN accuracy vs K (Exp A / B / C / no-adapt)
- `fig_p2_summary_bar.pdf`           — bar chart: Task 1 / 2A / 2B / 2C at K=8
- `fig_p2_atoms_comparison_K8.pdf`   — MNIST atoms vs SVHN-adapted atoms (K=8)
- `fig_p2_atoms_comparison_K4.pdf`   — same for K=4
- `fig_p2_pre_vs_adapted.pdf`        — pre-adaptation vs post-adaptation accuracy

---

## 5. Smoke Test (CPU, no GPU, ~5 min)

Verifies code correctness before submitting to the cluster:

```bash
# 2 epochs, K list = [1, 2] only (edit K_LIST in the scripts if needed),
# small batch, CPU mode
python train_baseline_part2.py  --epochs 2 --batch_size 32
python train_dcf_svhn_direct.py --epochs 2 --batch_size 32
python train_dcf_adaptation.py  --epochs_s1 2 --epochs_s2 2 --batch_size 32
python plot_results_part2.py
```

For a quick K_LIST override without editing the scripts:
```bash
# Temporarily use a short K list via sed (remember to revert before cluster run):
# sed -i 's/K_LIST = \[1, 2, 4, 6, 8\]/K_LIST = [1, 2]/' train_dcf_adaptation.py
```

---

## 6. Git Tracking — Add Everything After Training

Run from the **project root** (`ECE-662-Project/`):

```bash
# ── Baseline ──────────────────────────────────────────────────────────────────
git add part2/results_part2/baseline_mnist_best.pth
git add part2/results_part2/baseline_results.json

# ── Task 2A ───────────────────────────────────────────────────────────────────
git add part2/results_part2/svhn_direct_results.json

# ── Task 2B & 2C ──────────────────────────────────────────────────────────────
git add part2/results_part2/adaptation_results.json

# Model checkpoints (one per K per experiment)
for K in 1 2 4 6 8; do
  git add part2/results_part2/dcf_mnist_K${K}.pth
  git add part2/results_part2/dcf_svhn_expB_K${K}.pth
  git add part2/results_part2/dcf_svhn_expC_K${K}.pth
  git add part2/results_part2/atoms_mnist_K${K}.npy
  git add part2/results_part2/atoms_svhn_expB_K${K}.npy
done

# ── Plots ─────────────────────────────────────────────────────────────────────
git add part2/results_part2/fig_p2_accuracy_vs_K.pdf
git add part2/results_part2/fig_p2_summary_bar.pdf
git add part2/results_part2/fig_p2_atoms_comparison_K8.pdf
git add part2/results_part2/fig_p2_atoms_comparison_K4.pdf
git add part2/results_part2/fig_p2_pre_vs_adapted.pdf

git commit -m "Add Part 2 trained weights, JSON results, and domain adaptation plots"
git push
```

Pull on your local machine:
```bash
git pull
```

---

## 7. Recommended SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=ece662_part2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --output=part2_run_%j.log

conda activate ece662
cd /path/to/ECE-662-Project/part2

# Run baseline first (fast)
python train_baseline_part2.py --epochs 30 --gpu 0
echo "Baseline done"

# Run Task 2A and Task 2B/C in sequence (both use a lot of memory)
python train_dcf_svhn_direct.py --epochs 30 --gpu 0
echo "Task 2A done"

python train_dcf_adaptation.py --epochs_s1 30 --epochs_s2 15 --gpu 0
echo "Task 2B/C done"

python plot_results_part2.py --rep_K 8
echo "Plots done"
```

---

## 8. Expected Final Results Summary

| Method | Training | SVHN Test Acc. |
|--------|----------|----------------|
| Task 1: Baseline (standard AlexNet) | MNIST → SVHN (no adapt) | ~25–40% |
| Task 2A: DCF K=8 (upper bound) | SVHN directly | ~82–88% |
| Task 2B: Adapt atoms ψ_k, K=8 | MNIST → SVHN (adapt ψ_k) | ~60–72% |
| Task 2C: Adapt coeff. a_k, K=8 | MNIST → SVHN (adapt a_k) | ~32–50% |

The gap between Task 2B and 2C validates the paper's hypothesis:
**sharing a_k across domains preserves semantic structure**.
