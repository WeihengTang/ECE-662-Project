# Part 1 — Execution Instructions (GPU Cluster)

> Run all commands from the `part1/` directory unless otherwise noted.

---

## 0. Environment Setup

```bash
# Recommended: Python 3.9+, CUDA 11.x
conda create -n ece662 python=3.9 -y
conda activate ece662
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install scipy numpy matplotlib

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 1. Task 1 — PCA on AlexNet Filters

### Step 1a: Train Baseline AlexNet on MNIST

```bash
cd /path/to/ECE-662-Project/part1
python train_alexnet.py --epochs 30 --lr 0.01 --batch_size 128 --gpu 0
```

Expected output:
- `results/alexnet_mnist_best.pth`  (best model weights)
- `results/alexnet_mnist_log.json`  (per-epoch accuracy/loss log)

Expected best test accuracy: ~99.0–99.3%

### Step 1b: Run PCA Analysis (no retraining needed)

```bash
python run_pca.py
```

Expected output:
- `results/pca_results.json`  (accuracy & params vs K for K=1..9)

### Step 1c: Train DCF-AlexNet (learnable atoms) for each K

```bash
python train_dcf_alexnet.py --epochs 30 --lr 0.01 --batch_size 128 --gpu 0
```

Expected output:
- `results/dcf_results.json`

> Wall-clock estimate: ~9 × 30 epochs × ~60 s/epoch ≈ 4–5 hours on a single GPU.
> To run faster, reduce `--epochs 15` for a quick sanity check.

---

## 2. Task 2 — Fourier-Bessel Bases

### Step 2a: Train FB-AlexNet (fixed FB atoms) for each K

```bash
python train_fb_alexnet.py --epochs 30 --lr 0.01 --batch_size 128 --gpu 0
```

Expected output:
- `results/fb_results.json`
- `results/fb_atoms_K8.npy`  (8 FB atoms of shape 3×3, for visualisation)

---

## 3. Task 3 — Denoising Autoencoder

### Step 3a: Train baseline denoising autoencoder

```bash
python train_autoencoder.py --epochs 30 --lr 1e-3 --noise_std 0.3 --gpu 0
```

Expected output:
- `results/autoencoder_best.pth`
- `results/autoencoder_train_log.json`

### Step 3b: Train DCF and FB autoencoders, sweep K

```bash
python train_autoencoder_dcf.py --epochs 30 --lr 1e-3 --noise_std 0.3 \
                                 --gpu 0 --mode both
```

Expected output:
- `results/autoencoder_dcf_results.json`
- `results/autoencoder_fb_results.json`

> Tip: use `--mode dcf` or `--mode fb` to run only one variant.

---

## 4. Generate All Plots

Run this **after all training scripts have finished**:

```bash
python plot_results.py
```

Expected output (in `results/`):
- `fig1_accuracy_vs_K.pdf`         — Task 1: test accuracy vs K
- `fig2_params_vs_K.pdf`           — Task 1: parameters vs K
- `fig3_fb_dcf_accuracy_vs_K.pdf`  — Task 2: FB vs DCF accuracy
- `fig4_fb_atoms.pdf`              — Task 2: FB bases visualisation
- `fig5_dcf_atoms.pdf`             — Task 2: PCA/DCF atoms visualisation
- `fig6_psnr_vs_K.pdf`             — Task 3: PSNR vs K
- `fig7_explained_variance.pdf`    — (bonus) cumulative explained variance

---

## 5. Git Tracking (commit all results back to the repo)

Run these from the project root (`ECE-662-Project/`) after all training is done:

```bash
# Stage trained weights
git add part1/results/alexnet_mnist_best.pth
git add part1/results/autoencoder_best.pth

# Stage JSON result logs
git add part1/results/alexnet_mnist_log.json
git add part1/results/pca_results.json
git add part1/results/dcf_results.json
git add part1/results/fb_results.json
git add part1/results/autoencoder_train_log.json
git add part1/results/autoencoder_dcf_results.json
git add part1/results/autoencoder_fb_results.json

# Stage numpy arrays (atoms for visualisation)
git add part1/results/fb_atoms_K8.npy

# Stage generated plots
git add part1/results/fig1_accuracy_vs_K.pdf
git add part1/results/fig2_params_vs_K.pdf
git add part1/results/fig3_fb_dcf_accuracy_vs_K.pdf
git add part1/results/fig4_fb_atoms.pdf
git add part1/results/fig5_dcf_atoms.pdf
git add part1/results/fig6_psnr_vs_K.pdf
git add part1/results/fig7_explained_variance.pdf

git commit -m "Add Part 1 trained weights, JSON logs, and result plots"
git push
```

After pushing, pull on your local machine:
```bash
git pull
```

---

## 6. Quick Smoke Test (CPU, ~2 min, no GPU required)

To verify code correctness before running on the cluster:

```bash
# Quick test: 2 epochs, K=[1,2] only, small batch
python train_alexnet.py --epochs 2 --batch_size 32
python run_pca.py
python train_dcf_alexnet.py --epochs 2 --batch_size 32
python train_fb_alexnet.py --epochs 2 --batch_size 32
python train_autoencoder.py --epochs 2 --batch_size 32
python train_autoencoder_dcf.py --epochs 2 --batch_size 32 --mode both
python plot_results.py
```

> All scripts detect CUDA automatically and fall back to CPU.

---

## 7. Recommended SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=ece662_part1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=part1_run.log

conda activate ece662
cd /path/to/ECE-662-Project/part1

python train_alexnet.py           --epochs 30 --gpu 0 &
wait

python run_pca.py                            &
python train_dcf_alexnet.py       --epochs 30 --gpu 0 &
wait

python train_fb_alexnet.py        --epochs 30 --gpu 0  &
python train_autoencoder.py       --epochs 30 --gpu 0 &
wait

python train_autoencoder_dcf.py   --epochs 30 --gpu 0 --mode both &
wait

python plot_results.py
```
