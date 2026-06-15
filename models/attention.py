"""Multi-head attention modules.

Provides:
  - ``ScaledDotProductAttention`` — standard dot-product attention kernel.
  - ``MultiheadAttention``        — generic MHA with optional edge features,
    gating, and a pluggable attention kernel.
"""

import math
from collections.abc import Mapping
from typing import Optional

import torch
import torch.nn as nn
from torch import BoolTensor, Tensor
from torch.nn.functional import sigmoid

from models.utils import masked_softmax, merge_masks


class ScaledDotProductAttention(nn.Module):
    """Standard scaled dot-product attention (Vaswani et al., 2017).

    Computes ``softmax(Q K^T / sqrt(d_k)) V`` with optional additive bias
    and dropout on the attention weights.

    Args:
        dropout: Dropout probability applied to attention weights after softmax.
                 Defaults to ``0.0`` (disabled).
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        scale: float,
        mask: Optional[BoolTensor] = None,
        attn_bias: Optional[Tensor] = None,
        return_scores: bool = False,
    ) -> Tensor:
        """Compute attention weights.

        Args:
            q:             (B, H, Lq, D) query projections.
            k:             (B, H, Lk, D) key projections.
            scale:         Scaling divisor, typically ``sqrt(head_dim)``.
            mask:          Bool mask broadcast-compatible with (B, H, Lq, Lk).
                           ``True`` = block this position.
            attn_bias:     (B, Lq, Lk, H) additive pre-softmax bias.
            return_scores: If ``True``, also return the raw pre-softmax scores.

        Returns:
            Attention weight tensor (B, H, Lq, Lk), or a tuple
            ``(weights, scores)`` when *return_scores* is ``True``.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        if attn_bias is not None:
            scores = scores + attn_bias.permute(0, 3, 1, 2)
        scores = self.dropout(scores)
        weights = masked_softmax(scores, mask)
        if return_scores:
            return weights, scores
        return weights


class MultiheadAttention(nn.Module):
    """Generic multi-head attention with optional edge features and output gating.

    Supports both **self-attention** (single input *q*) and **cross-attention**
    (separate *q* and *k* / *v*).  Optionally incorporates per-pair edge
    features that bias attention scores and gate value aggregation.

    Args:
        embed_dim:      Primary embedding and output dimension.
        num_heads:      Number of attention heads.  Must divide *embed_dim*.
        attention:      Pluggable attention kernel instance.
        edge_embed_dim: Dimension of per-pair edge features.  ``0`` disables
                        edge support.  Must divide *num_heads*.
        q_dim:          Override for the query *input* dimension.  Defaults to
                        *embed_dim*.
        k_dim:          Override for the key input dimension.  Defaults to
                        *embed_dim*.
        v_dim:          Override for the value input dimension.  Defaults to
                        *embed_dim*.
        out_proj:       Whether to apply a final output linear projection.
        update_edges:   Whether to compute updated edge features from attention
                        scores and return them alongside the token output.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention: nn.Module,
        edge_embed_dim: int = 0,
        q_dim: Optional[int] = None,
        k_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        out_proj: bool = True,
        update_edges: bool = False,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        if edge_embed_dim % num_heads != 0:
            raise ValueError(
                f"edge_embed_dim ({edge_embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.out_proj = out_proj
        self.edge_embed_dim = edge_embed_dim
        self.edge_head_dim = edge_embed_dim // num_heads
        self.k_dim = k_dim or embed_dim
        self.v_dim = v_dim or embed_dim
        self.scale = math.sqrt(self.head_dim)
        self.update_edges = update_edges

        if q_dim is None:
            self.q_dim = self.embed_dim
        else:
            self.q_dim = q_dim
            assert self.q_dim == self.num_heads or out_proj, (
                "q_dim must equal num_heads when out_proj is False"
            )

        if isinstance(attention, Mapping):
            raise NotImplementedError("Pass an attention module instance, not a dict")
        self.attention = attention

        self.linear_q = nn.Linear(self.embed_dim, self.embed_dim)
        self.linear_k = nn.Linear(self.k_dim, self.embed_dim)
        self.linear_v = nn.Linear(self.v_dim, self.embed_dim)

        if self.edge_embed_dim > 0:
            self.linear_e = nn.Linear(self.edge_embed_dim, self.num_heads)
            self.linear_g = nn.Linear(self.edge_embed_dim, self.num_heads)
            self.linear_e_out = (
                nn.Linear(self.num_heads, self.edge_embed_dim)
                if self.update_edges
                else None
            )

        self.linear_out = nn.Linear(self.embed_dim, self.q_dim) if self.out_proj else None

    def _project(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Apply Q/K/V projections and reshape to (B, H, L, head_dim)."""
        shape = (k.shape[0], -1, self.num_heads, self.head_dim)
        return (
            self.linear_q(q).view(shape).transpose(1, 2),
            self.linear_k(k).view(shape).transpose(1, 2),
            self.linear_v(v).view(shape).transpose(1, 2),
        )

    def forward(
        self,
        q: Tensor,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        edges: Optional[Tensor] = None,
        q_mask: Optional[BoolTensor] = None,
        kv_mask: Optional[BoolTensor] = None,
        attn_mask: Optional[BoolTensor] = None,
        attn_bias: Optional[Tensor] = None,
    ) -> Tensor:
        """Multi-head attention forward pass.

        Args:
            q:         (B, Lq, embed_dim) query tokens.
            k:         (B, Lk, k_dim) key tokens.  ``None`` → self-attention.
            v:         (B, Lk, v_dim) value tokens.  ``None`` → defaults to *k*.
            edges:     (B, Lq, Lk, edge_embed_dim) per-pair edge features.
            q_mask:    (B, Lq) bool padding mask.  ``True`` = padded query.
            kv_mask:   (B, Lk) bool padding mask.  ``True`` = padded key/value.
            attn_mask: (B, Lq, Lk) explicit bool attention mask.
            attn_bias: (B, Lq, Lk, H) additive attention bias.

        Returns:
            (B, Lq, embed_dim) updated token features, or a tuple
            ``(tokens, edge_out)`` when *update_edges* is ``True``.
        """
        if k is None:
            k = q
            if kv_mask is None:
                kv_mask = q_mask
        v = v if v is not None else k

        b = q.shape[0]
        merged_mask = merge_masks(q_mask, kv_mask, attn_mask, q.shape, k.shape, q.device)
        q_proj, k_proj, v_proj = self._project(q, k, v)

        if edges is not None:
            e_bias = self.linear_e(edges)
            g = sigmoid(self.linear_g(edges))
            attn_bias = e_bias if attn_bias is None else attn_bias + e_bias

        attn_weights = self.attention(
            q_proj, k_proj, self.scale, merged_mask, attn_bias, self.update_edges
        )
        if self.update_edges:
            attn_weights, attn_scores = attn_weights

        if edges is not None:
            attn_weights = attn_weights * g.permute(0, 3, 1, 2)

        out = torch.matmul(attn_weights, v_proj)
        out = out.transpose(1, 2).contiguous().view(b, -1, self.embed_dim)

        edge_out = None
        if self.update_edges:
            edge_out = self.linear_e_out(attn_scores.permute(0, 2, 3, 1))

        if self.linear_out is not None:
            out = self.linear_out(out)

        return (out, edge_out) if edges is not None else out
