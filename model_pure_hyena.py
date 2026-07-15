"""
PureGeometricHyena
==================
Crystal property prediction using pure Geometric Hyena — no cross-gating.

Motivation
----------
CrossGatedHyena (model.py / model_attn.py) runs two parallel streams
(node + edge) and gates their deltas against each other.  This tests
whether all of that machinery is actually necessary, or whether a direct
Hyena long convolution applied to each atom's geometric neighbourhood
sequence is sufficient.

Architecture per layer
----------------------
For each destination atom i:
    1.  Gather neighbour features:  h_src[e] = h_v[src[e]]   (E, node_dim)
    2.  _build_sequences → (N, max_k, d_seq)
            d_seq = node_dim + num_rbf_pos + num_rbf_angle
        Elements encode WHO the neighbour is (h_v) + HOW FAR (RBF dist) +
        WHAT ANGLE relative to the nearest bond (Chebyshev angle RBF).
        Sorted by ascending bond distance — nearest bond first.
    3.  LongConv with distance-conditioned per-channel filter (Levels 2+4)
            FFT(h_filter) ⊙ FFT(seq)  →  (N, max_k, d_seq)
    4.  Masked-mean aggregate → (N, d_seq)
    5.  Linear projection → delta_v  (N, node_dim)
    6.  Self-gate:  delta_v ⊙ (1 + tanh(W · h_v))
            Starts near identity, never kills signal, allows amplification.
    7.  Pre-LN residual:  h_v ← h_v + delta_v

Global structure
----------------
    node embedding
    → L × GeometricHyenaLayer (Pre-LN residual)
    → HierarchicalPool: N atoms → ⌈N/4⌉ cluster super-atoms
                        → target-specific attention pool  (B, node_dim, 2)
    → lattice embedding
    → readout_bg  (band gap, eV)
    → readout_fe  (formation energy per atom, eV/atom)
    → (B, 2)

No edge features, no cross-gating, no dual streams.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt

from model import _build_sequences, LongConv, EmbeddingBlock, HierarchicalPool


# ─────────────────────────────────────────────────────────────────────────────
# Single pure Geometric Hyena layer
# ─────────────────────────────────────────────────────────────────────────────

class GeometricHyenaLayer(nn.Module):
    """
    One pure Geometric Hyena message-passing step.

    Sequence elements for atom i are [h_v[j] ∥ RBF(d_ij) ∥ angle_RBF(θ_ij)]
    for each neighbour j, sorted by distance.  The Hyena long convolution
    then mixes these elements with a distance-conditioned filter, capturing
    long-range inductive biases across the ordered neighbourhood.

    Parameters
    ----------
    node_dim      : hidden dimension of atom features
    num_rbf_pos   : RBF basis size for distance (filter input)
    num_rbf_angle : Chebyshev basis size for bond-angle encoding
    max_k         : neighbourhood sequence length (k nearest neighbours)
    filter_hidden : hidden size of the LongConv filter MLP
    cutoff        : RBF / angle distance cutoff (Å)
    """

    def __init__(
        self,
        node_dim:      int,
        num_rbf_pos:   int,
        num_rbf_angle: int,
        max_k:         int,
        filter_hidden: int,
        cutoff:        float,
    ):
        super().__init__()
        self.max_k         = max_k
        self.num_rbf_pos   = num_rbf_pos
        self.num_rbf_angle = num_rbf_angle
        self.cutoff        = cutoff

        # Total sequence element dimension
        d_seq = node_dim + num_rbf_pos + num_rbf_angle

        self.long_conv = LongConv(num_rbf_pos, d_seq, filter_hidden)
        self.out_proj  = nn.Linear(d_seq, node_dim)

        # 1+tanh self-gate keeps a gradient path h_v → delta_v at init
        # and allows per-channel amplification/suppression without saturation
        self.v_gate = nn.Linear(node_dim, node_dim, bias=False)

    def forward(
        self,
        h_v:        torch.Tensor,   # (N, node_dim)
        edge_index: torch.Tensor,   # (2, E)
        unit_vec:   torch.Tensor,   # (E, 3)  unit bond vectors
        edge_dist:  torch.Tensor,   # (E,)    raw distances (Å)
    ) -> torch.Tensor:               # (N, node_dim)  delta
        N   = h_v.shape[0]
        src = edge_index[0]

        # Neighbour atom features act as the sequence elements
        h_src = h_v[src]   # (E, node_dim) — one entry per directed bond

        seq, dist_rbf, mask, _, _ = _build_sequences(
            edge_index, h_src, unit_vec, edge_dist,
            N, self.max_k, self.num_rbf_pos, self.num_rbf_angle,
            self.cutoff, group_by='dst', exclude_self_loops=True,
        )
        # seq     : (N, max_k, node_dim + num_rbf_pos + num_rbf_angle)
        # dist_rbf: (N, max_k, num_rbf_pos)
        # mask    : (N, max_k)

        conv_out = self.long_conv(seq, dist_rbf, mask)   # (N, max_k, d_seq)

        # Masked mean over valid neighbours (avoid dividing by zero)
        k_cnt = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        agg   = (conv_out * mask.unsqueeze(-1).float()).sum(dim=1) / k_cnt

        delta_v = self.out_proj(agg)                               # (N, node_dim)
        delta_v = delta_v * (1.0 + torch.tanh(self.v_gate(h_v)))  # self-gate

        return delta_v


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class PureGeometricHyena(nn.Module):
    """
    Crystal property predictor: pure Geometric Hyena, no cross-gating.

    Predicts band gap (eV) and formation energy per atom (eV/atom) jointly
    via separate readout heads fed from target-specific attention pooling.

    Parameters
    ----------
    node_in_dim            : continuous node feature width (default 8)
    node_dim               : hidden dimension throughout all layers
    num_layers             : number of GeometricHyenaLayer blocks
    num_rbf_pos            : RBF basis size for bond distances
    num_rbf_angle          : Chebyshev basis size for bond angles
    max_k                  : neighbourhood sequence length per atom
    filter_hidden          : hidden units in LongConv filter MLP
    cutoff                 : distance cutoff (Å) for RBF
    dropout                : dropout probability in readout MLPs
    cat_embed_dim          : embedding width per categorical column
    gradient_checkpointing : recompute activations in backward pass
                             (~75 % less activation VRAM at ~25 % extra compute)
    """

    _NODE_CAT_VOCABS  = [94, 18, 7, 20]
    _GRAPH_CAT_VOCABS = [7,  16]

    def __init__(
        self,
        node_in_dim:            int   = 8,
        node_dim:               int   = 128,
        num_layers:             int   = 4,
        num_rbf_pos:            int   = 64,
        num_rbf_angle:          int   = 32,
        max_k:                  int   = 16,
        filter_hidden:          int   = 128,
        cutoff:                 float = 6.0,
        dropout:                float = 0.1,
        cat_embed_dim:          int   = 16,
        gradient_checkpointing: bool  = True,
    ):
        super().__init__()

        # ── Categorical node embeddings ───────────────────────────────────
        self.node_cat_embeds = nn.ModuleList([
            nn.Embedding(vocab, cat_embed_dim) for vocab in self._NODE_CAT_VOCABS
        ])
        node_cat_dim = cat_embed_dim * len(self._NODE_CAT_VOCABS)   # 16×4 = 64

        # ── Initial node projection  (cat + cont → node_dim) ─────────────
        self.node_embed = EmbeddingBlock(
            node_cat_dim + node_in_dim, node_dim * 2, node_dim
        )

        # ── Stack of pure Geometric Hyena layers ──────────────────────────
        self.layers = nn.ModuleList([
            GeometricHyenaLayer(
                node_dim      = node_dim,
                num_rbf_pos   = num_rbf_pos,
                num_rbf_angle = num_rbf_angle,
                max_k         = max_k,
                filter_hidden = filter_hidden,
                cutoff        = cutoff,
            )
            for _ in range(num_layers)
        ])

        # ── Pre-LN normalization (one per layer) ──────────────────────────
        self.node_norms = nn.ModuleList([
            nn.LayerNorm(node_dim) for _ in range(num_layers)
        ])

        # ── Lattice feature embedding ─────────────────────────────────────
        lat_embed_dim = 16
        self.graph_cat_embeds = nn.ModuleList([
            nn.Embedding(vocab, lat_embed_dim) for vocab in self._GRAPH_CAT_VOCABS
        ])
        self.graph_embed = nn.Sequential(
            nn.Linear(lat_embed_dim * len(self._GRAPH_CAT_VOCABS) + 6, node_dim),
            nn.SiLU(),
        )

        self.gradient_checkpointing = gradient_checkpointing

        # ── Hierarchical coarsening + attention pool ──────────────────────
        # N atoms → ⌈N/4⌉ cluster super-atoms → (B, node_dim, 1)
        self.hier_pool = HierarchicalPool(node_dim, num_targets=1, coarsen_factor=4)

        # ── Band-gap readout head ─────────────────────────────────────────
        self.readout_bg = nn.Sequential(
            nn.LayerNorm(node_dim + node_dim),
            nn.Linear(node_dim + node_dim, node_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim // 2),
            nn.SiLU(),
            nn.Linear(node_dim // 2, 1),
        )

    # ──────────────────────────────────────────────────────────────────────

    def _embed_node(self, x_cat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        cat_embs = torch.cat(
            [emb(x_cat[:, i]) for i, emb in enumerate(self.node_cat_embeds)],
            dim=-1,
        )
        return self.node_embed(torch.cat([cat_embs, x], dim=-1))   # (N, node_dim)

    def _embed_lattice(
        self, graph_cat: torch.Tensor, graph_attr: torch.Tensor
    ) -> torch.Tensor:
        cat_embs = [emb(graph_cat[:, i]) for i, emb in enumerate(self.graph_cat_embeds)]
        return self.graph_embed(torch.cat(cat_embs + [graph_attr], dim=-1))   # (B, node_dim)

    def forward(self, data) -> torch.Tensor:
        """
        data fields required:
          .x_cat, .x              — node categorical + continuous features
          .edge_index              — COO atom-graph connectivity
          .edge_unit_vec           — (E, 3) unit bond vectors
          .edge_raw_dist           — (E,) bond distances in Å
          .graph_cat, .graph_attr  — crystal-level lattice features
          .batch                   — atom → graph mapping (set by PyG DataLoader)

        Returns (B, 1): band gap in eV.
        """
        N          = data.x.shape[0]
        edge_index = data.edge_index
        unit_vec   = data.edge_unit_vec
        edge_dist  = data.edge_raw_dist

        # Step 1: embed atom features
        h_v = self._embed_node(data.x_cat, data.x)   # (N, node_dim)

        # Step 2: stacked Geometric Hyena layers with Pre-LN residual
        use_ckpt = self.gradient_checkpointing and self.training
        for norm_v, layer in zip(self.node_norms, self.layers):
            if use_ckpt:
                delta_v = grad_ckpt(
                    layer,
                    norm_v(h_v), edge_index, unit_vec, edge_dist,
                    use_reentrant=False,
                )
            else:
                delta_v = layer(norm_v(h_v), edge_index, unit_vec, edge_dist)
            h_v = h_v + delta_v

        # Step 3: hierarchical pool  N → ⌈N/4⌉ → (B, node_dim, 2)
        batch = (
            data.batch
            if hasattr(data, 'batch') and data.batch is not None
            else torch.zeros(N, dtype=torch.long, device=h_v.device)
        )
        num_graphs = int(batch.max().item()) + 1
        h_pool = self.hier_pool(h_v, batch, num_graphs)   # (B, node_dim, 2)

        # Step 4: lattice features
        h_lat = self._embed_lattice(data.graph_cat, data.graph_attr)   # (B, node_dim)

        # Step 5: band-gap readout
        return self.readout_bg(torch.cat([h_pool[..., 0], h_lat], dim=-1))    # (B, 1)
