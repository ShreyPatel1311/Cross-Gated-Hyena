"""
Crystal graph utilities: basis function encoding and line-graph construction.
"""

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Basis encodings
# ---------------------------------------------------------------------------

def rbf_encode(distances: torch.Tensor, num_rbf: int = 64, cutoff: float = 10.0) -> torch.Tensor:
    """
    Gaussian radial basis functions over [0, cutoff].
    distances: (E,) → output: (E, num_rbf)
    """
    centers = torch.linspace(0.0, cutoff, num_rbf, device=distances.device)
    sigma = cutoff / num_rbf
    return torch.exp(-((distances.unsqueeze(-1) - centers) ** 2) / (2.0 * sigma ** 2))


def angle_encode(cos_angles: torch.Tensor, num_basis: int = 32) -> torch.Tensor:
    """
    Chebyshev polynomial basis T_n(x) = cos(n * arccos(x)) for x ∈ [-1, 1].
    cos_angles: (...,) → output: (..., num_basis)
    """
    theta = torch.acos(cos_angles.clamp(-1.0 + 1e-6, 1.0 - 1e-6))  # (...,)
    n = torch.arange(num_basis, device=cos_angles.device, dtype=torch.float32)
    return torch.cos(n * theta.unsqueeze(-1))  # (..., num_basis)


# ---------------------------------------------------------------------------
# Line-graph construction (for ALIGNN)
# ---------------------------------------------------------------------------

def build_line_graph(
    edge_index: torch.Tensor,
    edge_vec: torch.Tensor,
    edge_dist: torch.Tensor,
    max_neighbors: int = 12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the ALIGNN line graph from the atomic crystal graph.

    Interpretation
    --------------
    Line-graph nodes  = directed bonds in the atomic graph  (E total)
    Line-graph edges  = bond-angle triplets centred on a shared source atom
                        e1 = (i→j), e2 = (i→k)  ⟹  lg_edge (e1, e2)
    Line-graph feat   = cosine of the angle ∠(j–i–k)

    Args
    ----
    edge_index    : (2, E) int64 – atom-graph src/dst indices
    edge_vec      : (E, 3) float – bond vectors from src to dst atom
    edge_dist     : (E,)   float – bond distances (Å)
    max_neighbors : maximum outgoing bonds per atom to keep (k-NN by distance)

    Returns
    -------
    lg_edge_index : (2, L) int64 – line-graph edge pairs (indices into atomic edges)
    lg_cos        : (L,)   float – cosine angle for each line-graph edge
    """
    src = edge_index[0]
    num_atoms = int(src.max().item()) + 1

    # Group outgoing edges per source atom
    groups: list[list[int]] = [[] for _ in range(num_atoms)]
    for e in range(edge_index.shape[1]):
        groups[int(src[e].item())].append(e)

    lg_src_list: list[int] = []
    lg_dst_list: list[int] = []
    cos_list: list[float] = []

    for atom_edges in groups:
        # Trim to k nearest by distance
        atom_edges = sorted(atom_edges, key=lambda e: edge_dist[e].item())
        atom_edges = atom_edges[:max_neighbors]
        k = len(atom_edges)
        if k < 2:
            continue

        vecs = edge_vec[atom_edges]                                    # (k, 3)
        norms = vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        unit_vecs = vecs / norms                                       # (k, 3)
        cos_mat = (unit_vecs @ unit_vecs.T).clamp(-1.0, 1.0)          # (k, k)

        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                lg_src_list.append(atom_edges[i])
                lg_dst_list.append(atom_edges[j])
                cos_list.append(float(cos_mat[i, j].item()))

    if not lg_src_list:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.float32)

    lg_edge_index = torch.tensor([lg_src_list, lg_dst_list], dtype=torch.long)
    lg_cos = torch.tensor(cos_list, dtype=torch.float32)
    return lg_edge_index, lg_cos


# ---------------------------------------------------------------------------
# Per-edge aggregated angle features (used by ComFormer)
# ---------------------------------------------------------------------------

def compute_edge_angle_feats(
    edge_index: torch.Tensor,
    edge_vec: torch.Tensor,
    edge_dist: torch.Tensor,
    max_neighbors: int = 12,
    num_basis: int = 32,
) -> torch.Tensor:
    """
    For each directed edge (i→j), aggregate Chebyshev-encoded angles with all
    other edges (i→k, k≠j) that share source atom i.

    This produces an invariant per-edge angular context vector used by
    ComFormer as an additional bias in transformer attention.

    Returns
    -------
    angle_feats : (E, num_basis) float – aggregated angle encoding per bond
    """
    src = edge_index[0]
    num_edges = int(edge_index.shape[1])
    num_atoms = int(src.max().item()) + 1

    groups: list[list[int]] = [[] for _ in range(num_atoms)]
    for e in range(num_edges):
        groups[int(src[e].item())].append(e)

    angle_feats = torch.zeros(num_edges, num_basis, dtype=torch.float32)

    for atom_edges in groups:
        atom_edges = sorted(atom_edges, key=lambda e: edge_dist[e].item())
        atom_edges = atom_edges[:max_neighbors]
        k = len(atom_edges)
        if k < 2:
            continue

        vecs = edge_vec[atom_edges]
        norms = vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        unit_vecs = vecs / norms
        cos_mat = (unit_vecs @ unit_vecs.T).clamp(-1.0, 1.0)   # (k, k)

        for idx, e_j in enumerate(atom_edges):
            # Angles between edge e_j and all other edges from atom i
            other_mask = [i for i in range(k) if i != idx]
            if not other_mask:
                continue
            cos_vals = cos_mat[idx, other_mask]          # (k-1,)
            enc = angle_encode(cos_vals, num_basis)      # (k-1, num_basis)
            angle_feats[e_j] = enc.mean(dim=0)

    return angle_feats
