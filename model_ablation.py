"""
model_ablation.py
=================
CrossGatedHyena with a swappable cross-gating mechanism for ablation.

Gating types
------------
  'elementwise' : delta_v = delta_v  ⊙  proj(scatter_mean(delta_e))
  'film'        : delta_v = γ ⊙ delta_v + β   (γ, β from other stream)
  'bilinear'    : delta_v = proj_left(delta_v)  ⊙  proj_right(scatter_mean(delta_e))
  'gru'         : delta_v = z ⊙ delta_v + (1−z) ⊙ r   (GRU-style blend)
  'swiglu'      : delta_v = proj(delta_v) ⊙ SiLU(proj(scatter_mean(delta_e)))

Everything else (LongConv, HierarchicalPool, residuals, readout) is
identical across all variants so results are directly comparable.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
from torch_scatter import scatter, scatter_softmax

from model import (
    Sin, fftconv, _build_sequences,
    LongConv, EmbeddingBlock, HierarchicalPool,
)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-gating modules — one per gating type
# ─────────────────────────────────────────────────────────────────────────────

class ElementwiseGate(nn.Module):
    """x1 ⊙ proj(x2)  — current model.py baseline."""
    def __init__(self, dim1: int, dim2: int):
        super().__init__()
        self.proj = nn.Linear(dim2, dim1, bias=False)

    def forward(self, x1, x2):
        return x1 * self.proj(x2)


class FiLMGate(nn.Module):
    """γ ⊙ x1 + β  where (γ, β) = Linear(x2).chunk(2)."""
    def __init__(self, dim1: int, dim2: int):
        super().__init__()
        self.proj = nn.Linear(dim2, 2 * dim1)

    def forward(self, x1, x2):
        out = self.proj(x2)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma * x1 + beta


class BilinearGate(nn.Module):
    """proj_left(x1) ⊙ proj_right(x2)  — both streams transformed before gating."""
    def __init__(self, dim1: int, dim2: int):
        super().__init__()
        self.left  = nn.Linear(dim1, dim1, bias=False)
        self.right = nn.Linear(dim2, dim1, bias=False)

    def forward(self, x1, x2):
        return self.left(x1) * self.right(x2)


class GRUGate(nn.Module):
    """z ⊙ x1 + (1−z) ⊙ r  — interpolation between old and new."""
    def __init__(self, dim1: int, dim2: int):
        super().__init__()
        self.r_proj = nn.Linear(dim2, dim1, bias=False)          # value from x2
        self.z_gate = nn.Linear(dim1 + dim2, dim1)               # interpolation gate

    def forward(self, x1, x2):
        r = self.r_proj(x2)
        z = torch.sigmoid(self.z_gate(torch.cat([x1, x2], dim=-1)))
        return z * x1 + (1.0 - z) * r


class SwiGLUGate(nn.Module):
    """proj(x1) ⊙ SiLU(proj(x2))  — LLM-style gated linear unit."""
    def __init__(self, dim1: int, dim2: int):
        super().__init__()
        self.gate = nn.Linear(dim1, dim1, bias=False)   # transforms x1
        self.cond = nn.Linear(dim2, dim1, bias=False)   # conditions via SiLU

    def forward(self, x1, x2):
        return self.gate(x1) * F.silu(self.cond(x2))


_GATE_CLASSES = {
    'elementwise': ElementwiseGate,
    'film':        FiLMGate,
    'bilinear':    BilinearGate,
    'gru':         GRUGate,
    'swiglu':      SwiGLUGate,
}


# ─────────────────────────────────────────────────────────────────────────────
# HyenaLayer with swappable cross-gating
# ─────────────────────────────────────────────────────────────────────────────

class HyenaLayer(nn.Module):
    def __init__(
        self,
        node_dim:      int,
        edge_dim:      int,
        num_rbf_pos:   int,
        num_rbf_angle: int,
        max_k:         int,
        filter_hidden: int,
        cutoff:        float,
        gating:        str = 'elementwise',
    ):
        super().__init__()
        assert gating in _GATE_CLASSES, f"Unknown gating '{gating}'. Choose from {list(_GATE_CLASSES)}"
        self.gating        = gating
        self.max_k         = max_k
        self.num_rbf_pos   = num_rbf_pos
        self.num_rbf_angle = num_rbf_angle
        self.cutoff        = cutoff

        d_seq = edge_dim + num_rbf_pos + num_rbf_angle

        self.nodeLongConv  = LongConv(num_rbf_pos, d_seq, filter_hidden)
        self.edgeLongConv  = LongConv(num_rbf_pos, d_seq, filter_hidden)
        self.node_out_proj = nn.Linear(d_seq, node_dim)
        self.edge_out_proj = nn.Linear(d_seq, edge_dim)

        GateCls = _GATE_CLASSES[gating]
        self.gate_v = GateCls(node_dim, edge_dim)   # edges gate nodes
        self.gate_e = GateCls(edge_dim, node_dim)   # nodes gate edges

        self.v_gate = nn.Linear(node_dim, node_dim, bias=False)   # self-gate

    def forward(self, h_v, h_e, edge_index, unit_vec, edge_dist):
        N   = h_v.shape[0]
        E   = edge_index.shape[1]
        src = edge_index[0]
        dst = edge_index[1]

        # ── Node stream ───────────────────────────────────────────────────
        node_seq, node_dist_rbf, node_mask, _, _ = _build_sequences(
            edge_index, h_e, unit_vec, edge_dist,
            N, self.max_k, self.num_rbf_pos, self.num_rbf_angle,
            self.cutoff, group_by='dst', exclude_self_loops=True,
        )
        node_conv = self.nodeLongConv(node_seq, node_dist_rbf, node_mask)
        k_cnt     = node_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        n_agg     = (node_conv * node_mask.unsqueeze(-1).float()).sum(dim=1) / k_cnt
        delta_v   = self.node_out_proj(n_agg)

        # ── Edge stream ───────────────────────────────────────────────────
        edge_seq, edge_dist_rbf, edge_mask, flat_idx, orig_idx = _build_sequences(
            edge_index, h_e, unit_vec, edge_dist,
            N, self.max_k, self.num_rbf_pos, self.num_rbf_angle,
            self.cutoff, group_by='src', exclude_self_loops=True,
        )
        edge_conv            = self.edgeLongConv(edge_seq, edge_dist_rbf, edge_mask)
        conv_flat            = edge_conv.reshape(N * self.max_k, -1)
        h_e_conv             = h_e.new_zeros(E, conv_flat.shape[-1])
        h_e_conv[orig_idx]   = conv_flat[flat_idx]
        delta_e              = self.edge_out_proj(h_e_conv)

        # ── Cross-gate ────────────────────────────────────────────────────
        x2_for_v = scatter(delta_e, dst, dim=0, dim_size=N, reduce='mean')
        delta_v  = self.gate_v(delta_v, x2_for_v)

        x2_for_e = delta_v[src]
        delta_e  = self.gate_e(delta_e, x2_for_e)

        # ── Self-gate ─────────────────────────────────────────────────────
        delta_v = delta_v * torch.sigmoid(self.v_gate(h_v))

        return delta_v, delta_e


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class CrossGatedHyena(nn.Module):
    _NODE_CAT_VOCABS  = [94, 18, 7, 20]
    _EDGE_CAT_VOCABS  = [7,  7,  7]
    _GRAPH_CAT_VOCABS = [7,  16]

    def __init__(
        self,
        node_in_dim:            int   = 8,
        edge_in_dim:            int   = 64,
        node_dim:               int   = 128,
        edge_dim:               int   = 128,
        num_layers:             int   = 4,
        num_rbf_pos:            int   = 64,
        num_rbf_angle:          int   = 32,
        max_k:                  int   = 16,
        filter_hidden:          int   = 128,
        cutoff:                 float = 6.0,
        dropout:                float = 0.1,
        cat_embed_dim:          int   = 16,
        gradient_checkpointing: bool  = True,
        gating:                 str   = 'elementwise',
    ):
        super().__init__()
        self.gating = gating

        self.node_cat_embeds = nn.ModuleList([
            nn.Embedding(v, cat_embed_dim) for v in self._NODE_CAT_VOCABS
        ])
        self.edge_cat_embeds = nn.ModuleList([
            nn.Embedding(v, cat_embed_dim) for v in self._EDGE_CAT_VOCABS
        ])
        node_cat_dim = cat_embed_dim * len(self._NODE_CAT_VOCABS)
        edge_cat_dim = cat_embed_dim * len(self._EDGE_CAT_VOCABS)

        self.node_embed = EmbeddingBlock(node_cat_dim + node_in_dim, node_dim * 2, node_dim)
        self.edge_embed = EmbeddingBlock(edge_cat_dim + edge_in_dim, edge_dim * 2, edge_dim)

        self.layers = nn.ModuleList([
            HyenaLayer(
                node_dim=node_dim, edge_dim=edge_dim,
                num_rbf_pos=num_rbf_pos, num_rbf_angle=num_rbf_angle,
                max_k=max_k, filter_hidden=filter_hidden,
                cutoff=cutoff, gating=gating,
            )
            for _ in range(num_layers)
        ])

        self.node_norms = nn.ModuleList([nn.LayerNorm(node_dim) for _ in range(num_layers)])
        self.edge_norms = nn.ModuleList([nn.LayerNorm(edge_dim) for _ in range(num_layers)])

        lat_embed_dim = 16
        self.graph_cat_embeds = nn.ModuleList([
            nn.Embedding(v, lat_embed_dim) for v in self._GRAPH_CAT_VOCABS
        ])
        self.graph_embed = nn.Sequential(
            nn.Linear(lat_embed_dim * len(self._GRAPH_CAT_VOCABS) + 6, node_dim),
            nn.SiLU(),
        )

        self.gradient_checkpointing = gradient_checkpointing
        self.hier_pool = HierarchicalPool(node_dim, num_targets=1)

        self.readout_bg = nn.Sequential(
            nn.LayerNorm(node_dim + node_dim),
            nn.Linear(node_dim + node_dim, node_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim // 2),
            nn.SiLU(),
            nn.Linear(node_dim // 2, 1),
        )

    def _embed_node(self, x_cat, x):
        cat = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.node_cat_embeds)], dim=-1)
        return self.node_embed(torch.cat([cat, x], dim=-1))

    def _embed_edge(self, edge_cat, edge_attr):
        cat = torch.cat([emb(edge_cat[:, i]) for i, emb in enumerate(self.edge_cat_embeds)], dim=-1)
        return self.edge_embed(torch.cat([cat, edge_attr], dim=-1))

    def _embed_lattice(self, graph_cat, graph_attr):
        cat = [emb(graph_cat[:, i]) for i, emb in enumerate(self.graph_cat_embeds)]
        return self.graph_embed(torch.cat(cat + [graph_attr], dim=-1))

    def forward(self, data):
        N          = data.x.shape[0]
        edge_index = data.edge_index
        unit_vec   = data.edge_unit_vec
        edge_dist  = data.edge_raw_dist

        h_v = self._embed_node(data.x_cat, data.x)
        h_e = self._embed_edge(data.edge_cat, data.edge_attr)

        use_ckpt = self.gradient_checkpointing and self.training
        for norm_v, norm_e, layer in zip(self.node_norms, self.edge_norms, self.layers):
            if use_ckpt:
                delta_v, delta_e = grad_ckpt(
                    layer, norm_v(h_v), norm_e(h_e),
                    edge_index, unit_vec, edge_dist,
                    use_reentrant=False,
                )
            else:
                delta_v, delta_e = layer(norm_v(h_v), norm_e(h_e), edge_index, unit_vec, edge_dist)
            h_v = h_v + delta_v
            h_e = h_e + delta_e

        batch      = data.batch if hasattr(data, 'batch') and data.batch is not None \
                     else torch.zeros(N, dtype=torch.long, device=h_v.device)
        num_graphs = int(batch.max().item()) + 1
        h_pool     = self.hier_pool(h_v, batch, num_graphs)
        h_lat      = self._embed_lattice(data.graph_cat, data.graph_attr)

        return self.readout_bg(torch.cat([h_pool[..., 0], h_lat], dim=-1))
