"""
ablation.py
===========
Runs a fast cross-gating ablation: 5 variants × 10 epochs × 10 K samples.

Usage (in Colab, after cloning the repo):
    !python /content/Files/ablation.py

Results are printed as a table and saved to /content/ablation_results.csv
"""

import os, sys, time, csv
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# ── paths — auto-detect Colab vs local ────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))

if os.path.exists("/content/Files"):                          # Colab runtime
    REPO_DIR   = "/content/Files"
    H5_PATH    = "/content/Files/graphs_data.h5"
    STATS_PATH = "/content/Files/node_stats.pt"
    OUT_CSV    = f"/content/ablation_results_seed{SEED}.csv"
else:                                                          # local machine
    REPO_DIR   = _here
    H5_PATH    = os.path.join(_here, "graphs_data.h5")
    STATS_PATH = os.path.join(_here, "node_stats.pt")
    OUT_CSV    = os.path.join(_here, f"ablation_results_seed{SEED}.csv")

CSV_PATH = os.path.join(REPO_DIR, "materials_tabular.csv")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# ── ablation config ────────────────────────────────────────────────────────
GATING_TYPES  = ['elementwise', 'film', 'bilinear', 'gru', 'swiglu', 'outer_product', 'cross_attn']
EPOCHS        = 10
MAX_IDS       = 10_000
SEED          = 0       # change to 1, 2, … for each repeat run
EDGE_BUDGET   = 25_000
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
ACCUM_STEPS   = 4
NODE_DIM      = 128
EDGE_DIM      = 128
NUM_LAYERS    = 4
NUM_RBF_POS   = 64
NUM_RBF_ANGLE = 32
MAX_K         = 16
FILTER_HIDDEN = 128

# ── setup ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
p      = torch.cuda.get_device_properties(0) if device.type == "cuda" else None
cap    = (p.major, p.minor) if p else (0, 0)
AMP_DTYPE       = torch.bfloat16 if cap[0] >= 8 else torch.float16
USE_AMP         = device.type == "cuda"
USE_GRAD_SCALER = (AMP_DTYPE == torch.float16) and USE_AMP
import random, numpy as np
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
print(f"Device: {device}  AMP: {AMP_DTYPE if USE_AMP else 'off'}  Seed: {SEED}")

from dataset import CrossGatedHyenaDataset, EdgeBudgetBatchSampler
from model_ablation import CrossGatedHyena

# ── dataset (shared across all runs) ──────────────────────────────────────
print(f"\nLoading {MAX_IDS:,} samples …")
stats  = torch.load(STATS_PATH)
ds     = CrossGatedHyenaDataset(H5_PATH, CSV_PATH, num_rbf=NUM_RBF_POS,
                                 max_ids=MAX_IDS, node_stats=stats, augment=False)
n      = len(ds)
n_tr   = int(0.80 * n);  n_val = int(0.10 * n);  n_te = n - n_tr - n_val
gen    = torch.Generator().manual_seed(42)
tr_ds, val_ds, te_ds = random_split(ds, [n_tr, n_val, n_te], generator=gen)

tr_samp  = EdgeBudgetBatchSampler(tr_ds,  EDGE_BUDGET, shuffle=True)
val_samp = EdgeBudgetBatchSampler(val_ds, EDGE_BUDGET, shuffle=False)
te_samp  = EdgeBudgetBatchSampler(te_ds,  EDGE_BUDGET, shuffle=False)

_ldr = dict(num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)
train_loader = DataLoader(tr_ds, batch_sampler=tr_samp, **_ldr)
val_loader   = DataLoader(val_ds, batch_sampler=val_samp, **_ldr)
test_loader  = DataLoader(te_ds,  batch_sampler=te_samp, **_ldr)
print(f"Train {n_tr:,}  Val {n_val:,}  Test {n_te:,}")


# ── helpers ────────────────────────────────────────────────────────────────
criterion = nn.L1Loss()

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    mae = mse = n = 0
    for b in loader:
        b = b.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            pred = model(b).float().view(-1)
        tgt     = b.y.float().view(-1)
        pred_eV = torch.expm1(pred)
        mae += (pred_eV - tgt).abs().sum().item()
        mse += ((pred_eV - tgt) ** 2).sum().item()
        n   += tgt.numel()
    return mae / n, (mse / n) ** 0.5


def train_one_variant(gating: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  Gating: {gating.upper()}")
    print(f"{'='*60}")

    model = CrossGatedHyena(
        node_in_dim=8, edge_in_dim=NUM_RBF_POS,
        node_dim=NODE_DIM, edge_dim=EDGE_DIM,
        num_layers=NUM_LAYERS, num_rbf_pos=NUM_RBF_POS,
        num_rbf_angle=NUM_RBF_ANGLE, max_k=MAX_K,
        filter_hidden=FILTER_HIDDEN, cutoff=6.0,
        gradient_checkpointing=True, gating=gating,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optim     = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optim, T_max=EPOCHS, eta_min=LR * 1e-3)
    scaler    = torch.amp.GradScaler(enabled=USE_GRAD_SCALER)

    best_val_mae = float('inf')
    history = []

    for epoch in range(1, EPOCHS + 1):
        tr_samp.set_epoch(epoch)
        ds.train()
        model.train()
        t0 = time.perf_counter()
        total_loss = n_step = 0
        optim.zero_grad(set_to_none=True)

        bar = tqdm(train_loader, desc=f"  Ep {epoch:2d}/{EPOCHS}", leave=False)
        for step, b in enumerate(bar):
            b = b.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                pred = model(b).view(-1)
                tgt  = torch.log1p(b.y.float().view(-1).clamp(min=0))
                loss = criterion(pred, tgt) / ACCUM_STEPS

            if not torch.isfinite(loss):
                print(f"  Non-finite loss at ep {epoch} step {step} — skipping batch")
                optim.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            total_loss += loss.item() * ACCUM_STEPS
            n_step     += 1

            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

        ds.eval()
        scheduler.step()
        val_mae, val_rmse = evaluate(model, val_loader)
        elapsed = time.perf_counter() - t0
        best_val_mae = min(best_val_mae, val_mae)

        history.append({'epoch': epoch, 'val_mae': val_mae, 'val_rmse': val_rmse})
        star = ' ★' if val_mae <= best_val_mae else ''
        print(f"  Ep {epoch:2d}  loss={total_loss/max(n_step,1):.4f}  "
              f"val_MAE={val_mae:.4f} eV  lr={scheduler.get_last_lr()[0]:.1e}  "
              f"{elapsed:.0f}s{star}")

    test_mae, test_rmse = evaluate(model, test_loader)
    print(f"\n  ► Best val MAE : {best_val_mae:.4f} eV")
    print(f"  ► Test MAE     : {test_mae:.4f} eV")
    print(f"  ► Test RMSE    : {test_rmse:.4f} eV")

    del model
    torch.cuda.empty_cache()

    return {
        'gating'       : gating,
        'n_params'     : n_params,
        'best_val_mae' : best_val_mae,
        'test_mae'     : test_mae,
        'test_rmse'    : test_rmse,
        'history'      : history,
    }


# ── run ablation ───────────────────────────────────────────────────────────
all_results = []
for g in GATING_TYPES:
    r = train_one_variant(g)
    all_results.append(r)

# ── summary table ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  ABLATION SUMMARY  ({EPOCHS} epochs, {MAX_IDS:,} samples)")
print(f"{'='*60}")
print(f"  {'Gating':<14}  {'Params':>10}  {'Best Val MAE':>13}  {'Test MAE':>10}  {'Test RMSE':>11}")
print(f"  {'-'*14}  {'-'*10}  {'-'*13}  {'-'*10}  {'-'*11}")

all_results.sort(key=lambda x: x['best_val_mae'])
for r in all_results:
    print(f"  {r['gating']:<14}  {r['n_params']:>10,}  "
          f"{r['best_val_mae']:>12.4f} eV  "
          f"{r['test_mae']:>9.4f} eV  "
          f"{r['test_rmse']:>10.4f} eV")

# ── save to CSV ────────────────────────────────────────────────────────────
with open(OUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['gating', 'n_params', 'best_val_mae', 'test_mae', 'test_rmse'])
    writer.writeheader()
    for r in all_results:
        writer.writerow({k: r[k] for k in writer.fieldnames})

print(f"\nSaved → {OUT_CSV}")
