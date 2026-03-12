import os

# ── Shared resource paths ────────────────────────────────────────────────────
# bessel.npy lives in the cloned DCFNet-Pytorch repo
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

path_to_bessel = os.path.join(_ROOT, 'DCFNet-Pytorch', 'bessel.npy')

# Part-1 module path (dcf_layer.py, fb.py, models/)
PART1_DIR = os.path.join(_ROOT, 'part1')

# Results directory for Part 2
RESULTS_DIR = os.path.join(_HERE, 'results_part2')
os.makedirs(RESULTS_DIR, exist_ok=True)
