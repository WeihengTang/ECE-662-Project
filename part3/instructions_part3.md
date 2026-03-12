# Part 3 — Execution Instructions (GPU Cluster)

> Run all commands from the `part3/` directory.
> Part 3 imports models and layers from `../part1/`, so `part1/` must be present.

---

## 0. Environment Setup

Same environment as Parts 1 & 2 (if already activated, skip this):

```bash
conda activate ece662
cd /path/to/ECE-662-Project/part3

# Quick sanity check
python -c "from dataset_blur import get_blur_task_loaders; print('dataset_blur OK')"
python -c "from cl_utils import compute_psnr; print('cl_utils OK')"
```

---

## 1. Baseline — Naive Sequential Training (Catastrophic Forgetting)

```bash
python train_cl_baseline.py \
    --epochs_per_task 20 --lr 1e-3 --batch_size 128 --gpu 0 --seed 42
```

What this does:
1. Creates a standard Conv autoencoder.
2. Trains sequentially: Task 0 (Gaussian) → Task 1 (Motion) → Task 2 (Defocus).
3. After each phase, evaluates PSNR on **all** tasks → reveals forgetting.

Expected outputs:
- `results_part3/baseline_cl_results.json`
- `results_part3/baseline_model_after_t{0,1,2}.pth`

Expected result: PSNR on Task 0 drops sharply after training Task 1 and 2.

> Wall-clock: ~3 tasks × 20 epochs × ~30 s/epoch ≈ 30 min on 1 GPU.

---

## 2. DCF Continual-Learning via Coefficient Swapping

```bash
python train_cl_dcf.py \
    --num_bases 6 --epochs_per_task 20 --lr 1e-3 \
    --batch_size 128 --gpu 0 --seed 42
```

What this does:
1. Task 0: trains both ψ_k (atoms) and a_k (coefficients) on Gaussian blur.
2. Task 1 & 2: **freezes ψ_k**, re-initialises a_k, trains only task-specific
   params on the new blur type.
3. Evaluates each task by swapping the correct a_k back into the model.

Expected outputs:
- `results_part3/dcf_cl_results.json`         — full PSNR matrix + memory
- `results_part3/dcf_shared_atoms.pth`        — shared ψ_k (frozen after T0)
- `results_part3/dcf_task{0,1,2}_coeff.pth`   — per-task coefficient dicts

Expected result: PSNR on Task 0 stays **constant** even after Task 1 and 2
(zero forgetting by construction).

> Wall-clock: ~3 tasks × 20 epochs × ~30 s/epoch ≈ 30 min on 1 GPU.

---

## 3. Generate All Plots

Run **after** both training scripts have finished:

```bash
python plot_cl_results.py
```

Expected outputs in `results_part3/`:
- `fig_p3_forgetting_curves.pdf`   — PSNR vs training phase for all tasks
- `fig_p3_memory_footprint.pdf`    — stacked bar: baseline vs DCF-CL memory
- `fig_p3_psnr_heatmaps.pdf`       — side-by-side PSNR matrices
- `fig_p3_blur_examples.pdf`       — PSF kernels + example blurred digits

---

## 4. Smoke Test (CPU, ~5 min)

```bash
python train_cl_baseline.py  --epochs_per_task 2 --batch_size 32
python train_cl_dcf.py       --num_bases 4 --epochs_per_task 2 --batch_size 32
python plot_cl_results.py
```

---

## 5. Git Tracking

From the project root (`ECE-662-Project/`):

```bash
# ── JSON logs ──────────────────────────────────────────────────────────────────
git add part3/results_part3/baseline_cl_results.json
git add part3/results_part3/dcf_cl_results.json

# ── Model checkpoints ─────────────────────────────────────────────────────────
git add part3/results_part3/baseline_model_after_t0.pth
git add part3/results_part3/baseline_model_after_t1.pth
git add part3/results_part3/baseline_model_after_t2.pth
git add part3/results_part3/dcf_shared_atoms.pth
git add part3/results_part3/dcf_task0_coeff.pth
git add part3/results_part3/dcf_task1_coeff.pth
git add part3/results_part3/dcf_task2_coeff.pth

# ── Plots ──────────────────────────────────────────────────────────────────────
git add part3/results_part3/fig_p3_forgetting_curves.pdf
git add part3/results_part3/fig_p3_memory_footprint.pdf
git add part3/results_part3/fig_p3_psnr_heatmaps.pdf
git add part3/results_part3/fig_p3_blur_examples.pdf

git commit -m "Add Part 3 continual deblurring results and plots"
git push
```

Pull on your local machine:
```bash
git pull
```

---

## 6. Recommended SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=ece662_part3
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=part3_run_%j.log

conda activate ece662
cd /path/to/ECE-662-Project/part3

python train_cl_baseline.py --epochs_per_task 20 --gpu 0
echo "Baseline done"

python train_cl_dcf.py --num_bases 6 --epochs_per_task 20 --gpu 0
echo "DCF-CL done"

python plot_cl_results.py
echo "Plots done"
```
