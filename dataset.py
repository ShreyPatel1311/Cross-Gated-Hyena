"""
CrossGatedHyenaDataset
======================
Dedicated PyG dataset for the CrossGatedHyena model.

Why a separate dataset from ALIGNNDataset?
-------------------------------------------
ALIGNNDataset builds the line graph (angle graph) on every __getitem__ call.
That is O(k²) edges per atom — for max_k=12 it is 12×11=132 line-graph edges
per atom, which completely dominates loading time when you are NOT using ALIGNN.
CrossGatedHyena builds its own per-atom sequences inside _build_sequences at
forward time, so none of that precomputation is needed here.

This dataset:
  • Skips line-graph construction entirely          → ~3–10× faster __getitem__
  • Never loads dead features (bond_type, eneg)    → less H5 I/O
  • Drops x_lin[:,1] (oxidation_state = 1.0)       → 8 instead of 9 node dims
  • Returns a lean Data object with exactly the    → no wasted memory per batch
    fields CrossGatedHyena.forward() accesses

Fields in each Data item
------------------------
  x_cat         (N, 4)        node categorical indices
  x             (N, 8)        node continuous features
  edge_cat      (E, 3)        edge categorical (periodic image shifts)
  edge_attr     (E, num_rbf)  Gaussian-RBF distance encoding
  edge_index    (2, E)        COO atom-graph connectivity
  edge_unit_vec (E, 3)        unit bond vectors  (equivariant)
  edge_raw_dist (E,)          raw bond distances in Å
  graph_cat     (1, 2)        crystal system + Bravais class
  graph_attr    (1, 6)        normalised lattice lengths + angles in radians
  y             (1,)          band-gap target (eV), if csv_path supplied

Usage
-----
  ds = CrossGatedHyenaDataset("graphs_data.h5", "materials_tabular.csv")
  loader = DataLoader(ds, batch_size=32, num_workers=4)
"""

from __future__ import annotations

import os
from typing import Optional

import h5py
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graph_utils import rbf_encode


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_raw(h5: h5py.File, mid: str):
    """
    Read only the fields CrossGatedHyena needs.
    Dead features (edge_bond_type, edge_eneg) are intentionally skipped.
    """
    g = h5[mid]
    return (
        torch.tensor(g["x_cat"][:],         dtype=torch.long),    # (N, 4)
        torch.tensor(g["x_lin"][:],         dtype=torch.float32), # (N, 5)
        torch.tensor(g["x_log"][:],         dtype=torch.float32), # (N, 4)
        torch.tensor(g["edge_index"][:],    dtype=torch.long),    # (2, E)
        torch.tensor(g["edge_distance"][:], dtype=torch.float32), # (E,)
        torch.tensor(g["edge_vec"][:],      dtype=torch.float32), # (E, 3)
        torch.tensor(g["edge_cat"][:],      dtype=torch.long),    # (E, 3)
        torch.tensor(g["graph_abc"][:],     dtype=torch.float32).squeeze(0),  # (3,)
        torch.tensor(g["graph_angles"][:],  dtype=torch.float32).squeeze(0),  # (3,)
        torch.tensor(g["graph_cat"][:],     dtype=torch.long).squeeze(0),     # (2,)
    )

def _node_feats(x_lin: torch.Tensor, x_log: torch.Tensor) -> torch.Tensor:
    """
    (N, 9) → (N, 8): drop x_lin[:,1] (oxidation_state, always 1.0).
    Kept columns: electronegativity(0), atomic_radius(2), atomic_number(3), charge(4)
    + log1p of the four x_log columns.
    """
    lin_clean = torch.cat([x_lin[:, :1], x_lin[:, 2:]], dim=-1)  # (N, 4)
    log_safe  = torch.log1p(x_log.abs())                          # (N, 4)
    return torch.cat([lin_clean, log_safe], dim=-1)               # (N, 8)


def _edge_feats(edge_dist: torch.Tensor, num_rbf: int, cutoff: float) -> torch.Tensor:
    """(E,) → (E, num_rbf): Gaussian-RBF of bond distance, only real-signal edge feature."""
    return rbf_encode(edge_dist, num_rbf=num_rbf, cutoff=cutoff)


def _graph_feats(abc: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """(3,), (3,) → (6,): normalised lattice lengths + angles in radians."""
    return torch.cat([abc / 10.0, angles * (torch.pi / 180.0)], dim=-1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CrossGatedHyenaDataset(Dataset):
    """
    Crystal graph dataset tailored for CrossGatedHyena.

    Parameters
    ----------
    h5_path       : path to graphs_data.h5
    csv_path      : path to materials_tabular.csv (both band_gap and
                    formation_energy_per_atom columns required).
                    Pass None for inference without targets.
    num_rbf       : Gaussian-RBF basis size for bond-distance encoding
    cutoff        : distance cutoff (Å) for RBF centres
    max_ids       : cap the dataset size (useful for quick testing)
    augment       : if True, apply inversion-symmetry augmentation in training
                    mode (flip edge unit vectors with 50 % probability).
                    Call .eval() to disable during validation/test.

    y shape
    -------
    data.y is (2,): [band_gap_eV, formation_energy_per_atom_eV]
    """

    def __init__(
        self,
        h5_path:  str,
        csv_path: Optional[str] = None,
        num_rbf:  int   = 64,
        cutoff:   float = 10.0,
        max_ids:  Optional[int] = None,
        node_stats: Optional[dict[str, torch.Tensor]] = None,
        augment:  bool  = False,
    ):
        self.h5_path    = h5_path
        self.num_rbf    = num_rbf
        self.cutoff     = cutoff
        self.node_stats = node_stats
        self.augment    = augment
        self.training   = True   # set to False for val/test to disable augmentation

        # ── Build ID list and target map ──────────────────────────────────
        with h5py.File(h5_path, "r") as f:
            h5_ids = set(f.keys())

        # targets[mid] = (band_gap, formation_energy_per_atom)
        self.targets: dict[str, tuple[float, float]] = {}
        if csv_path is not None:
            df = pd.read_csv(csv_path)[
                ["material_id", "band_gap", "formation_energy_per_atom"]
            ].dropna()
            self.targets = {
                row.material_id: (float(row.band_gap), float(row.formation_energy_per_atom))
                for row in df.itertuples(index=False)
            }
            valid_ids = sorted(h5_ids & set(self.targets.keys()))
        else:
            valid_ids = sorted(h5_ids)

        if max_ids is not None:
            valid_ids = valid_ids[:max_ids]
        self.ids = valid_ids

    def train(self) -> "CrossGatedHyenaDataset":
        self.training = True
        return self

    def eval(self) -> "CrossGatedHyenaDataset":
        self.training = False
        return self

    # ──────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Data:
        mid = self.ids[idx]

        with h5py.File(self.h5_path, "r") as f:
            (x_cat, x_lin, x_log,
             edge_index, edge_dist, edge_vec, edge_cat,
             graph_abc, graph_angles, graph_cat) = _load_raw(f, mid)

        data = Data(
            x_cat         = x_cat,                                    # (N, 4)
            x             = _node_feats(x_lin, x_log),                # (N, 8)
            edge_cat      = edge_cat,                                  # (E, 3)
            edge_attr     = _edge_feats(edge_dist, self.num_rbf,       # (E, num_rbf)
                                        self.cutoff),
            edge_index    = edge_index,                                # (2, E)
            edge_unit_vec = F.normalize(edge_vec, dim=-1),            # (E, 3)
            edge_raw_dist = edge_dist,                                 # (E,)
            graph_cat     = graph_cat.unsqueeze(0),                    # (1, 2) → (B, 2)
            graph_attr    = _graph_feats(graph_abc,                    # (1, 6) → (B, 6)
                                         graph_angles).unsqueeze(0),
        )
        if self.node_stats is not None:
            mean = self.node_stats["mean"].to(data.x.device)
            std  = self.node_stats["std"].to(data.x.device)
            data.x = (data.x - mean) / std

        # Change 6: inversion-symmetry augmentation (valid for centrosymmetric crystals)
        if self.augment and self.training:
            if torch.rand(1).item() > 0.5:
                data.edge_unit_vec = -data.edge_unit_vec

        if mid in self.targets:
            bg, fe = self.targets[mid]
            data.y = torch.tensor([bg, fe], dtype=torch.float32)  # (2,)

        return data

    # ──────────────────────────────────────────────────────────────────────
    # Edge count cache (used by EdgeBudgetBatchSampler)
    # ──────────────────────────────────────────────────────────────────────

    @property
    def edge_counts(self) -> list[int]:
        """
        Number of edges per graph, lazily read from H5 (metadata only — fast).
        Result is cached after the first call.
        """
        if not hasattr(self, "_edge_counts"):
            print(f"Pre-computing edge counts for {len(self.ids):,} graphs …", flush=True)
            with h5py.File(self.h5_path, "r") as f:
                self._edge_counts = [
                    int(f[mid]["edge_index"].shape[1]) for mid in self.ids
                ]
            avg = sum(self._edge_counts) // len(self._edge_counts)
            print(f"  Done — avg {avg} edges/graph.", flush=True)
        return self._edge_counts

    # ──────────────────────────────────────────────────────────────────────
    # Convenience: pre-built DataLoaders
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def make_loaders(
        cls,
        h5_path:      str,
        csv_path:     str,
        num_rbf:      int            = 64,
        cutoff:       float          = 10.0,
        max_ids:      Optional[int]  = None,
        batch_size:   int            = 32,
        edge_budget:  Optional[int]  = None,
        num_workers:  int            = 4,
        train_frac:   float          = 0.80,
        val_frac:     float          = 0.10,
        seed:         int            = 42,
        augment:      bool           = False,
    ) -> tuple[DataLoader, DataLoader, DataLoader, "CrossGatedHyenaDataset"]:
        """
        Build train / val / test DataLoaders in one call.

        Parameters
        ----------
        edge_budget : if set, use EdgeBudgetBatchSampler instead of a fixed
                      batch_size.  Each batch contains as many graphs as fit
                      within this total edge count.  Prevents OOM spikes from
                      variable-size graphs and gives better GPU utilisation on
                      small crystals.  Typical value: 20_000 – 40_000.
        augment     : enable inversion-symmetry augmentation for the training
                      split only.  val/test always run without augmentation.

        Returns (train_loader, val_loader, test_loader, full_dataset).
        """
        ds    = cls(h5_path, csv_path, num_rbf=num_rbf, cutoff=cutoff,
                    max_ids=max_ids, augment=augment)
        n     = len(ds)
        n_tr  = int(train_frac * n)
        n_val = int(val_frac  * n)
        n_te  = n - n_tr - n_val

        gen              = torch.Generator().manual_seed(seed)
        tr_ds, val_ds, te_ds = random_split(ds, [n_tr, n_val, n_te], generator=gen)
        # val/test must not augment — switch the shared base dataset to eval mode
        # before creating val/test loaders; caller should call ds.train() again
        # before the training loop.
        # (All three Subsets share the same base ds instance.)

        ldr_kw = dict(num_workers=num_workers, pin_memory=True,
                      persistent_workers=(num_workers > 0),
                      prefetch_factor=2 if num_workers > 0 else None)

        if edge_budget is not None:
            tr_sampler  = EdgeBudgetBatchSampler(tr_ds,  edge_budget, shuffle=True,  seed=seed)
            val_sampler = EdgeBudgetBatchSampler(val_ds, edge_budget, shuffle=False, seed=seed)
            te_sampler  = EdgeBudgetBatchSampler(te_ds,  edge_budget, shuffle=False, seed=seed)
            return (
                DataLoader(tr_ds,  batch_sampler=tr_sampler,  **ldr_kw),
                DataLoader(val_ds, batch_sampler=val_sampler, **ldr_kw),
                DataLoader(te_ds,  batch_sampler=te_sampler,  **ldr_kw),
                ds,
            )

        ldr_kw["batch_size"] = batch_size
        return (
            DataLoader(tr_ds,  shuffle=True,  **ldr_kw),
            DataLoader(val_ds, shuffle=False, **ldr_kw),
            DataLoader(te_ds,  shuffle=False, **ldr_kw),
            ds,
        )


# ---------------------------------------------------------------------------
# Edge-budget batch sampler
# ---------------------------------------------------------------------------

class EdgeBudgetBatchSampler(torch.utils.data.BatchSampler):
    """
    Batches graphs by total edge count rather than graph count.

    Problem solved
    --------------
    With a fixed batch_size=32, a batch that happens to include one 22 K-edge
    crystal uses 18× more VRAM than a batch of 1 200-edge crystals.  This
    causes random OOM crashes and wastes GPU cycles on padding.

    Solution
    --------
    Keep accumulating graphs into a batch until `edge_budget` is reached, then
    start a new batch.  Result: every batch uses roughly the same VRAM.
    Small crystals get larger batches (better GPU parallelism), large crystals
    get their own batch (no OOM).

    Parameters
    ----------
    dataset     : CrossGatedHyenaDataset or torch.utils.data.Subset of one
    edge_budget : max total edges per batch (typical: 20_000 – 40_000)
    shuffle     : shuffle graph order before grouping (set True for train)
    seed        : random seed for shuffling
    """

    def __init__(
        self,
        dataset,
        edge_budget: int  = 25_000,
        shuffle:     bool = True,
        seed:        int  = 42,
    ):
        self.edge_budget = edge_budget
        self.shuffle     = shuffle
        self.seed        = seed

        # Resolve Subset → base dataset + local index mapping
        if isinstance(dataset, torch.utils.data.Subset):
            base_ds      = dataset.dataset
            subset_local = list(dataset.indices)   # global idx in base_ds
        else:
            base_ds      = dataset
            subset_local = list(range(len(dataset)))

        all_ec       = base_ds.edge_counts              # triggers lazy load
        self._ec     = [all_ec[i] for i in subset_local]
        self._n      = len(self._ec)
        self._build(list(range(self._n)))

    # ------------------------------------------------------------------

    def _build(self, order: list[int]) -> None:
        """Group indices in `order` into batches that stay within edge_budget."""
        self._batches: list[list[int]] = []
        batch, total = [], 0
        for idx in order:
            ec = self._ec[idx]
            if ec > self.edge_budget:
                # Oversized graph: flush current batch then give it its own
                if batch:
                    self._batches.append(batch)
                self._batches.append([idx])
                batch, total = [], 0
            elif batch and total + ec > self.edge_budget:
                self._batches.append(batch)
                batch, total = [idx], ec
            else:
                batch.append(idx)
                total += ec
        if batch:
            self._batches.append(batch)

    def set_epoch(self, epoch: int) -> None:
        """
        Re-shuffle and re-group for a new epoch so batch compositions change.
        Call this at the start of each training epoch:
            sampler.set_epoch(epoch)
        """
        gen   = torch.Generator().manual_seed(self.seed + epoch)
        order = torch.randperm(self._n, generator=gen).tolist()
        self._build(order)

    def __iter__(self):
        if self.shuffle:
            g     = torch.Generator().manual_seed(self.seed)
            order = torch.randperm(len(self._batches), generator=g).tolist()
            for i in order:
                yield self._batches[i]
        else:
            yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


# ---------------------------------------------------------------------------
# Standalone verification / timing
# ---------------------------------------------------------------------------

def _verify(h5_path: str, csv_path: str, n_samples: int = 200):
    """Load n_samples items and report shapes, timing, and field stats."""
    import time

    print(f"=== CrossGatedHyenaDataset verification ({n_samples} samples) ===")
    ds = CrossGatedHyenaDataset(h5_path, csv_path, max_ids=n_samples)
    print(f"Dataset size : {len(ds)}")

    t0   = time.time()
    item = ds[0]
    t1   = time.time()

    print(f"\nSingle item fields:")
    for k, v in item:
        if hasattr(v, "shape"):
            print(f"  {k:20s}: {str(tuple(v.shape)):20s} {v.dtype}")
    print(f"\nFirst __getitem__ time: {(t1-t0)*1e3:.1f} ms")

    # Time 20 items
    t0 = time.time()
    for i in range(20):
        _ = ds[i]
    avg_ms = (time.time() - t0) / 20 * 1e3
    print(f"Average __getitem__ time: {avg_ms:.1f} ms")

    # Check shapes are consistent
    nodes, edges, bands = [], [], []
    for i in range(min(50, len(ds))):
        d = ds[i]
        nodes.append(d.x.shape[0])
        edges.append(d.edge_attr.shape[0])
        if d.y is not None:
            bands.append(d.y.item())

    import statistics
    print(f"\nStats over {min(50, len(ds))} items:")
    print(f"  Nodes : min={min(nodes)}  max={max(nodes)}  mean={statistics.mean(nodes):.1f}")
    print(f"  Edges : min={min(edges)}  max={max(edges)}  mean={statistics.mean(edges):.1f}")
    if bands:
        print(f"  Band gap : min={min(bands):.3f}  max={max(bands):.3f}  "
              f"mean={statistics.mean(bands):.3f} eV")

    # DataLoader batch test
    print(f"\nDataLoader batch test (batch_size=4) …")
    from torch_geometric.loader import DataLoader as GLoader
    loader = GLoader(ds, batch_size=4, shuffle=False)
    batch  = next(iter(loader))
    print(f"  batch.x            : {tuple(batch.x.shape)}")
    print(f"  batch.edge_attr    : {tuple(batch.edge_attr.shape)}")
    print(f"  batch.edge_unit_vec: {tuple(batch.edge_unit_vec.shape)}")
    print(f"  batch.edge_raw_dist: {tuple(batch.edge_raw_dist.shape)}")
    print(f"  batch.graph_cat    : {tuple(batch.graph_cat.shape)}")
    print(f"  batch.graph_attr   : {tuple(batch.graph_attr.shape)}")
    if batch.y is not None:
        print(f"  batch.y            : {tuple(batch.y.shape)}")

    print("\nAll checks passed ✓")
    return ds


if __name__ == "__main__":
    import sys, os
    # Allow running from any directory
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)

    H5  = os.path.join(root, "graphs_data.h5")
    CSV = os.path.join(root, "materials_tabular.csv")

    ds = _verify(H5, CSV, n_samples=500)

    print("\n=== make_loaders convenience method ===")
    train_loader, val_loader, test_loader, full_ds = CrossGatedHyenaDataset.make_loaders(
        H5, CSV,
        max_ids    = 500,
        batch_size = 16,
        num_workers= 0,   # 0 for inline test
    )
    print(f"Train batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")
    print(f"Test  batches : {len(test_loader)}")

    b = next(iter(train_loader))
    print(f"\nTrain batch x: {b.x.shape}  edge_attr: {b.edge_attr.shape}")

