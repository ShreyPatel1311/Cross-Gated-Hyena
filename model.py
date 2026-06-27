from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
from torch_scatter import scatter
from graph_utils import rbf_encode, angle_encode


# ─────────────────────────────────────────────────────────────────────────────
# Level 1 + 2 + 3  ──  Neighbourhood sequence builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_sequences(
    edge_index:         torch.Tensor,   # (2, E)
    edge_feat:          torch.Tensor,   # (E, d_edge)
    unit_vec:           torch.Tensor,   # (E, 3)
    edge_dist:          torch.Tensor,   # (E,)
    N_atoms:            int,
    max_k:              int,
    num_rbf_pos:        int,
    num_rbf_angle:      int,
    cutoff:             float,
    group_by:           str  = 'dst',
    exclude_self_loops: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build per-atom neighbourhood sequences for Hyena long convolution.

    Level 1  per-atom, distance-sorted  (no cross-crystal contamination)
    Level 2  returns dist_rbf for filter conditioning on real Å values
    Level 3  enriched element = [edge_feat | dist_rbf | angle_rbf_from_nearest]

    group_by='dst'  incoming edges per destination atom  →  node stream
    group_by='src'  outgoing edges per source atom       →  edge stream

    exclude_self_loops  (default True)
        Self-loops (src == dst, periodic images of the same atom) are kept in
        edge_index for cross-gating message passing but excluded from the Hyena
        sequence.  They carry no angular diversity (all map to the same lattice
        translation direction) and pollute the nearest-bond angle reference,
        so removing them makes the sequence strictly more informative.
        Edges excluded from the sequence get zero long-conv update (the model
        falls back to their original embedding via the residual in HyenaLayer).

    Returns
    -------
    seq          (N_atoms, max_k, d_seq)
    dist_rbf_seq (N_atoms, max_k, num_rbf_pos)
    mask         (N_atoms, max_k)
    flat_idx     (E_valid,)  position in flattened (N×max_k,)
    orig_idx     (E_valid,)  original edge index in 0..E-1
    """
    d_edge = edge_feat.shape[-1]
    d_seq  = d_edge + num_rbf_pos + num_rbf_angle
    dev    = edge_feat.device
    E      = edge_index.shape[1]

    grp_full = edge_index[1] if group_by == 'dst' else edge_index[0]

    # ── Optionally strip self-loops from the sequence ─────────────────────
    if exclude_self_loops:
        keep        = edge_index[0] != edge_index[1]          # (E,) bool
        global_idx  = torch.where(keep)[0]                    # original edge indices kept
        grp         = grp_full[global_idx]
        dist_w      = edge_dist[global_idx]
        feat_w      = edge_feat[global_idx]
        uvec_w      = unit_vec[global_idx]
        E_w         = global_idx.shape[0]
    else:
        global_idx  = torch.arange(E, device=dev)
        grp, dist_w, feat_w, uvec_w, E_w = grp_full, edge_dist, edge_feat, unit_vec, E

    # ── Sort by (atom-group, distance) within working set ────────────────
    sort_key = grp.float() * 1e7 + dist_w
    s_idx    = torch.argsort(sort_key)                         # into working set

    grp_s  = grp[s_idx]
    dist_s = dist_w[s_idx]
    feat_s = feat_w[s_idx]
    uvec_s = uvec_w[s_idx]

    counts  = torch.bincount(grp_s, minlength=N_atoms)
    offsets = F.pad(counts.cumsum(0), (1, 0))

    row_w    = torch.arange(E_w, device=dev) - offsets[grp_s]
    valid    = row_w < max_k

    # orig_idx maps sequence positions → original (E,) edge space
    orig_idx = global_idx[s_idx[valid]]
    flat_idx = grp_s[valid] * max_k + row_w[valid]

    # ── Level 2: RBF-encode actual distances ─────────────────────────────
    dist_rbf_v = rbf_encode(dist_s[valid], num_rbf=num_rbf_pos, cutoff=cutoff)

    # ── Level 3: angle vs nearest bond in same group ──────────────────────
    g_valid      = grp_s[valid]
    nearest_uvec = uvec_s[offsets[g_valid]]                    # uvec of nearest bond
    cos_angles   = (uvec_s[valid] * nearest_uvec).sum(-1).clamp(-1.0, 1.0)
    angle_rbf_v  = angle_encode(cos_angles, num_rbf_angle)

    seq_elem = torch.cat([feat_s[valid], dist_rbf_v, angle_rbf_v], dim=-1)

    # ── Pack into (N_atoms, max_k, …) padded tensors ─────────────────────
    seq_out      = edge_feat.new_zeros(N_atoms * max_k, d_seq)
    dist_rbf_out = edge_feat.new_zeros(N_atoms * max_k, num_rbf_pos)
    mask_out     = torch.zeros(N_atoms * max_k, dtype=torch.bool, device=dev)

    seq_out[flat_idx]      = seq_elem
    dist_rbf_out[flat_idx] = dist_rbf_v
    mask_out[flat_idx]     = True

    return (
        seq_out.reshape(N_atoms, max_k, d_seq),
        dist_rbf_out.reshape(N_atoms, max_k, num_rbf_pos),
        mask_out.reshape(N_atoms, max_k),
        flat_idx,
        orig_idx,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Level 2 + 4  ──  LongConv with distance-conditioned per-channel filter
# ─────────────────────────────────────────────────────────────────────────────

class LongConv(nn.Module):
    """
    Hyena long convolution — Level 2 + 4.

    Level 2  filter MLP input = RBF(actual distance in Å), not rank index
    Level 4  filter MLP outputs seq_dim values (one per feature channel)

    out = IFFT( FFT(h, dim=1) ⊙ FFT(seq, dim=1) )
    """

    def __init__(self, num_rbf_pos: int, seq_dim: int, filter_hidden: int = 64):
        super().__init__()
        self.filter_mlp = nn.Sequential(
            nn.Linear(num_rbf_pos,   filter_hidden),
            nn.SiLU(),
            nn.Linear(filter_hidden, filter_hidden),
            nn.SiLU(),
            nn.Linear(filter_hidden, seq_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        seq:      torch.Tensor,   # (N, k, seq_dim)
        dist_rbf: torch.Tensor,   # (N, k, num_rbf)
        mask:     torch.Tensor,   # (N, k)
    ) -> torch.Tensor:
        _, k, _ = seq.shape
        m = mask.float().unsqueeze(-1)

        h   = self.filter_mlp(dist_rbf) * m
        seq = seq * m

        H   = torch.fft.rfft(h,   n=k, dim=1)
        V   = torch.fft.rfft(seq, n=k, dim=1)
        out = torch.fft.irfft(H * V, n=k, dim=1)
        return out * m


# ─────────────────────────────────────────────────────────────────────────────
# Shared building blocks
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingBlock(nn.Module):
    """Two-layer MLP: (N, in_dim) → SiLU → (N, out_dim)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim,     hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(u)))


class ElementWiseMultiplication(nn.Module):
    """
    Cross-modal gating: x1 ⊙ proj(x2_aligned).

    Resolves two mismatches:
      sequence length  (N vs E)  via scatter / gather
      feature dimension (dim1 vs dim2) via linear projection on x2
    """

    def __init__(
        self,
        embed1: nn.Module,
        embed2: nn.Module,
        dim1:   int,
        dim2:   int,
    ):
        super().__init__()
        self.embed1 = embed1
        self.embed2 = embed2
        self.proj = nn.Linear(dim2, dim1, bias=False) if dim1 != dim2 else nn.Identity()

    def forward(
        self,
        x1:         torch.Tensor,
        x2:         torch.Tensor,
        edge_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src, dst = (edge_index[0], edge_index[1]) if edge_index is not None else (None, None)

        if x1.shape[0] == x2.shape[0]:
            x2_seq = x2
        elif edge_index is not None:
            # Determine direction by comparing sizes: N_nodes < N_edges in all crystals.
            # Using x1.shape[0] instead of dst.max().item() avoids a torch.compile graph break.
            if x1.shape[0] < x2.shape[0]:
                # x1 = nodes (N, d), x2 = edges (E, d) → aggregate edges → nodes
                x2_seq = scatter(x2, dst, dim=0, dim_size=x1.shape[0], reduce="mean")
            else:
                # x1 = edges (E, d), x2 = nodes (N, d) → broadcast src node to each edge
                x2_seq = x2[src]
        else:
            x2_seq = x2.mean(0, keepdim=True).expand(x1.shape[0], -1)

        return x1 * self.proj(x2_seq)


# ─────────────────────────────────────────────────────────────────────────────
# HyenaLayer  ──  single cross-gated block, all four levels
# ─────────────────────────────────────────────────────────────────────────────

class HyenaLayer(nn.Module):
    """
    Single Geometric Hyena cross-gating block (all 4 levels).

    Operates on ALREADY-EMBEDDED features — does NOT do initial feature
    projection.  Designed to be stacked inside CrossGatedHyena.

    Returns pure DELTA tensors (changes, not full features) so the
    caller can apply a clean additive residual:
        h_v = h_v + delta_v
        h_e = h_e + delta_e

    Data flow
    ---------
    Node stream  (incoming edges per atom, group_by='dst')
        _build_sequences → (N, k, d_seq) enriched
        LongConv → masked mean → node_out_proj → delta_v  (N, node_dim)

    Edge stream  (outgoing edges per atom, group_by='src')
        _build_sequences → (N, k, d_seq) enriched
        LongConv → unpack to edges → edge_out_proj → delta_e  (E, edge_dim)

    Cross-modal gating on the deltas
        delta_v_gated = delta_v ⊙ proj( scatter_mean(delta_e → atoms) )
        delta_e_gated = delta_e ⊙ proj( delta_v[src] )

    Parameters
    ----------
    node_dim      : hidden node feature dimension
    edge_dim      : hidden edge feature dimension
    num_rbf_pos   : RBF basis size  (Level 2 filter input dimension)
    num_rbf_angle : Chebyshev basis size for angle encoding  (Level 3)
    max_k         : max neighbours per atom in Hyena sequence  (Level 1)
    filter_hidden : hidden size of LongConv filter MLP
    cutoff        : RBF distance cutoff (Å)
    """

    def __init__(
        self,
        node_dim:      int   = 64,
        edge_dim:      int   = 64,
        num_rbf_pos:   int   = 32,
        num_rbf_angle: int   = 16,
        max_k:         int   = 12,
        filter_hidden: int   = 64,
        cutoff:        float = 10.0,
    ):
        super().__init__()

        self.max_k         = max_k
        self.num_rbf_pos   = num_rbf_pos
        self.num_rbf_angle = num_rbf_angle
        self.cutoff        = cutoff

        d_seq = edge_dim + num_rbf_pos + num_rbf_angle   # Level 3 enriched dim

        # Level 2 + 4: long convolutions
        self.nodeLongConv = LongConv(num_rbf_pos, d_seq, filter_hidden)
        self.edgeLongConv = LongConv(num_rbf_pos, d_seq, filter_hidden)

        # Project conv output → stream-specific hidden dim
        self.node_out_proj = nn.Linear(d_seq, node_dim)
        self.edge_out_proj = nn.Linear(d_seq, edge_dim)

        # Cross-modal gating (gates the DELTAS)
        self.elemNode = ElementWiseMultiplication(
            self.nodeLongConv, self.edgeLongConv, dim1=node_dim, dim2=edge_dim
        )
        self.elemEdge = ElementWiseMultiplication(
            self.edgeLongConv, self.nodeLongConv, dim1=edge_dim, dim2=node_dim
        )

        # Self-gate: node current state h_v modulates how much of delta_v to apply.
        # This creates a direct gradient path from delta_v → h_v → node_norms,
        # which would otherwise be severed because both sequence streams use h_e only.
        self.v_gate = nn.Linear(node_dim, node_dim, bias=False)

    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        h_v:        torch.Tensor,   # (N, node_dim)  pre-embedded node features
        h_e:        torch.Tensor,   # (E, edge_dim)  pre-embedded edge features
        edge_index: torch.Tensor,   # (2, E)
        unit_vec:   torch.Tensor,   # (E, 3)  equivariant unit bond vectors
        edge_dist:  torch.Tensor,   # (E,)    raw bond distances (Å)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        delta_v : (N, node_dim)  node feature delta
        delta_e : (E, edge_dim)  edge feature delta
        """
        N = h_v.shape[0]
        E = edge_index.shape[1]

        # ── Node stream ───────────────────────────────────────────────────
        node_seq, node_dist_rbf, node_mask, _, _ = _build_sequences(
            edge_index, h_e, unit_vec, edge_dist,
            N, self.max_k, self.num_rbf_pos, self.num_rbf_angle,
            self.cutoff, group_by='dst', exclude_self_loops=True,
        )
        node_conv = self.nodeLongConv(node_seq, node_dist_rbf, node_mask)

        k_cnt = node_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        n_agg = (node_conv * node_mask.unsqueeze(-1).float()).sum(dim=1) / k_cnt
        delta_v = self.node_out_proj(n_agg)   # (N, node_dim)

        # ── Edge stream ───────────────────────────────────────────────────
        edge_seq, edge_dist_rbf, edge_mask, flat_idx, orig_idx = _build_sequences(
            edge_index, h_e, unit_vec, edge_dist,
            N, self.max_k, self.num_rbf_pos, self.num_rbf_angle,
            self.cutoff, group_by='src', exclude_self_loops=True,
        )
        edge_conv  = self.edgeLongConv(edge_seq, edge_dist_rbf, edge_mask)
        conv_flat  = edge_conv.reshape(N * self.max_k, -1)
        h_e_conv   = h_e.new_zeros(E, conv_flat.shape[-1])
        h_e_conv[orig_idx] = conv_flat[flat_idx]
        delta_e = self.edge_out_proj(h_e_conv)   # (E, edge_dim)

        # ── Cross-gate the deltas (not the full features) ─────────────────
        delta_v = self.elemNode(delta_v, delta_e, edge_index)   # (N, node_dim)
        delta_e = self.elemEdge(delta_e, delta_v, edge_index)   # (E, edge_dim)

        # ── Self-gate: modulate node delta by the node's current state ────
        # h_v carries information from the previous layer (or the initial
        # embedding).  The sigmoid gate in [0,1] controls per-channel how
        # much of the long-conv update to absorb, conditioned on what the
        # node already "knows".  This is the only path that connects h_v
        # (and therefore node_norms in CrossGatedHyena) to the loss.
        delta_v = delta_v * torch.sigmoid(self.v_gate(h_v))

        return delta_v, delta_e


# ─────────────────────────────────────────────────────────────────────────────
# CrossGatedHyena  ──  full model stacking N HyenaLayer blocks
# ─────────────────────────────────────────────────────────────────────────────

class CrossGatedHyena(nn.Module):
    """
    Full crystal band-gap model: stacks multiple HyenaLayer blocks with
    Pre-LN residual connections.

    Architecture
    ------------
    1. Initial projection  (raw dims → hidden dims, once only)
         node_embed : (N, node_in_dim) → (N, node_dim)
         edge_embed : (E, edge_in_dim) → (E, edge_dim)

    2. For each of num_layers blocks:
         Pre-LayerNorm → HyenaLayer → additive residual
         h_v = h_v + HyenaLayer( LN(h_v), LN(h_e), … ).delta_v
         h_e = h_e + HyenaLayer( … ).delta_e

    3. Global mean pool over atoms (per crystal in batch)

    4. Lattice feature embedding  (graph_cat + graph_attr)

    5. Readout MLP → band gap (eV)

    Pre-LN convention  (Xiong et al., 2020)
    ----------------------------------------
    Normalising BEFORE each sub-layer (not after) gives more stable
    gradients across depth, critical when stacking many Hyena blocks.

    Parameters
    ----------
    node_in_dim   : continuous node feature width  (data.x  — 8 dims after dead-col drop)
    edge_in_dim   : continuous edge feature width  (data.edge_attr — num_rbf dims)
    node_dim      : hidden node feature dimension throughout all layers
    edge_dim      : hidden edge feature dimension throughout all layers
    num_layers    : number of HyenaLayer blocks to stack
    num_rbf_pos   : RBF basis size for distance (Level 2)
    num_rbf_angle : Chebyshev basis size for angle (Level 3)
    max_k                  : k-NN sequence length per atom (Level 1)
    filter_hidden          : hidden size of the LongConv filter MLP
    cutoff                 : distance cutoff for RBF encoding (Å)
    dropout                : dropout in the readout MLP
    cat_embed_dim          : embedding size for each categorical feature column
    gradient_checkpointing : if True, recompute layer activations during backward
                             instead of storing them.  Cuts activation VRAM by ~75%
                             for 4 layers at ~25% extra compute.  Zero accuracy impact.

    Categorical features embedded internally
    ----------------------------------------
    Node  x_cat  (N, 4): atomic_idx(0-93), valence_group(0-16),
                          crystal_field(0-6), period_group(0-19)
    Edge  edge_cat (E, 3): periodic image shift per axis, each 0-6
    These are looked up via per-column nn.Embedding layers and concatenated
    with the continuous features before projection — no caller changes needed.
    """

    # Vocabulary sizes confirmed from full-dataset analysis
    _NODE_CAT_VOCABS  = [94, 18, 7, 20]   # x_cat cols: max values [93,17,6,19] → +1
    _EDGE_CAT_VOCABS  = [7,  7,  7]        # edge_cat columns (periodic shifts 0-6)
    _GRAPH_CAT_VOCABS = [7,  16]            # space_group_category, crystallographic_class

    def __init__(
        self,
        node_in_dim:            int   = 8,    # continuous only: x_lin(4) + x_log(4)
        edge_in_dim:            int   = 64,   # continuous only: RBF(64)
        node_dim:               int   = 64,
        edge_dim:               int   = 64,
        num_layers:             int   = 4,
        num_rbf_pos:            int   = 32,
        num_rbf_angle:          int   = 16,
        max_k:                  int   = 12,
        filter_hidden:          int   = 64,
        cutoff:                 float = 10.0,
        dropout:                float = 0.1,
        cat_embed_dim:          int   = 16,
        gradient_checkpointing: bool  = True,
    ):
        super().__init__()

        # ── Categorical embeddings for node x_cat and edge edge_cat ──────
        # Each column gets its own Embedding table; outputs are concatenated
        # with the continuous features before the MLP projection.
        self.node_cat_embeds = nn.ModuleList([
            nn.Embedding(vocab, cat_embed_dim) for vocab in self._NODE_CAT_VOCABS
        ])
        self.edge_cat_embeds = nn.ModuleList([
            nn.Embedding(vocab, cat_embed_dim) for vocab in self._EDGE_CAT_VOCABS
        ])
        node_cat_dim = cat_embed_dim * len(self._NODE_CAT_VOCABS)   # 16×4 = 64
        edge_cat_dim = cat_embed_dim * len(self._EDGE_CAT_VOCABS)   # 16×3 = 48

        # ── Initial feature projections (cat + cont → hidden) ────────────
        self.node_embed = EmbeddingBlock(node_cat_dim + node_in_dim, node_dim * 2, node_dim)
        self.edge_embed = EmbeddingBlock(edge_cat_dim + edge_in_dim, edge_dim * 2, edge_dim)

        # ── Stack of HyenaLayer blocks ────────────────────────────────────
        self.layers = nn.ModuleList([
            HyenaLayer(
                node_dim      = node_dim,
                edge_dim      = edge_dim,
                num_rbf_pos   = num_rbf_pos,
                num_rbf_angle = num_rbf_angle,
                max_k         = max_k,
                filter_hidden = filter_hidden,
                cutoff        = cutoff,
            )
            for _ in range(num_layers)
        ])

        # ── Pre-LN normalization (one pair per layer) ─────────────────────
        self.node_norms = nn.ModuleList([nn.LayerNorm(node_dim) for _ in range(num_layers)])
        self.edge_norms = nn.ModuleList([nn.LayerNorm(edge_dim) for _ in range(num_layers)])

        # ── Graph-level lattice feature embedding ─────────────────────────
        lat_embed_dim = 16
        self.graph_cat_embeds = nn.ModuleList([
            nn.Embedding(vocab, lat_embed_dim)
            for vocab in self._GRAPH_CAT_VOCABS
        ])
        self.graph_embed = nn.Sequential(
            nn.Linear(lat_embed_dim * len(self._GRAPH_CAT_VOCABS) + 6, node_dim),
            nn.SiLU(),
        )

        self.gradient_checkpointing = gradient_checkpointing

        # ── Readout: global node pool + lattice feat → band gap ───────────
        self.readout = nn.Sequential(
            nn.LayerNorm(node_dim + node_dim),
            nn.Linear(node_dim + node_dim, node_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim // 2),
            nn.SiLU(),
            nn.Linear(node_dim // 2, 1),
        )

    # ──────────────────────────────────────────────────────────────────────

    def _embed_lattice(
        self,
        graph_cat:  torch.Tensor,   # (B, 2) int
        graph_attr: torch.Tensor,   # (B, 6) float
    ) -> torch.Tensor:
        cat_embs = [emb(graph_cat[:, i]) for i, emb in enumerate(self.graph_cat_embeds)]
        return self.graph_embed(torch.cat(cat_embs + [graph_attr], dim=-1))   # (B, node_dim)

    def _embed_node(self, x_cat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Embed x_cat columns, concatenate with continuous x, project."""
        cat_embs = torch.cat(
            [emb(x_cat[:, i]) for i, emb in enumerate(self.node_cat_embeds)],
            dim=-1,
        )                                           # (N, node_cat_dim)
        return self.node_embed(torch.cat([cat_embs, x], dim=-1))  # (N, node_dim)

    def _embed_edge(self, edge_cat: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Embed edge_cat columns, concatenate with RBF edge_attr, project."""
        cat_embs = torch.cat(
            [emb(edge_cat[:, i]) for i, emb in enumerate(self.edge_cat_embeds)],
            dim=-1,
        )                                           # (E, edge_cat_dim)
        return self.edge_embed(torch.cat([cat_embs, edge_attr], dim=-1))  # (E, edge_dim)

    def forward(self, data) -> torch.Tensor:
        """
        data requires:
          .x_cat, .x              — node categorical + continuous features
          .edge_cat, .edge_attr   — edge categorical + RBF distance features
          .edge_index, .edge_unit_vec, .edge_raw_dist
          .graph_cat, .graph_attr
          .batch  (set by PyG DataLoader)

        Returns (B, 1) predicted band gaps in eV.
        """
        N          = data.x.shape[0]
        edge_index = data.edge_index
        unit_vec   = data.edge_unit_vec
        edge_dist  = data.edge_raw_dist

        # ── Step 1: embed categorical + continuous features ───────────────
        h_v = self._embed_node(data.x_cat, data.x)          # (N, node_dim)
        h_e = self._embed_edge(data.edge_cat, data.edge_attr)  # (E, edge_dim)

        # ── Step 2: stacked Hyena blocks with Pre-LN residual ─────────────
        use_ckpt = self.gradient_checkpointing and self.training
        for norm_v, norm_e, layer in zip(self.node_norms, self.edge_norms, self.layers):
            if use_ckpt:
                # Recompute layer activations during backward instead of storing
                # them. Mathematically identical gradients; ~75% less activation
                # VRAM for 4 layers at ~25% extra compute cost.
                delta_v, delta_e = grad_ckpt(
                    layer,
                    norm_v(h_v), norm_e(h_e),
                    edge_index, unit_vec, edge_dist,
                    use_reentrant=False,
                )
            else:
                delta_v, delta_e = layer(
                    norm_v(h_v), norm_e(h_e),
                    edge_index, unit_vec, edge_dist,
                )
            h_v = h_v + delta_v
            h_e = h_e + delta_e

        # ── Step 3: global mean pool per crystal ──────────────────────────
        batch = (
            data.batch
            if hasattr(data, "batch") and data.batch is not None
            else torch.zeros(N, dtype=torch.long, device=h_v.device)
        )
        num_graphs = int(batch.max().item()) + 1
        h_pool = scatter(h_v, batch, dim=0, dim_size=num_graphs, reduce="mean")  # (B, node_dim)

        # ── Step 4: lattice features ──────────────────────────────────────
        h_lat = self._embed_lattice(data.graph_cat, data.graph_attr)   # (B, node_dim)

        # ── Step 5: predict ───────────────────────────────────────────────
        return self.readout(torch.cat([h_pool, h_lat], dim=-1))        # (B, 1)

