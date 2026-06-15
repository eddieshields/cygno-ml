"""GPT-2 + Normformer-style transformer encoder.

Implements a stack of encoder layers based on the Normformer architecture
(Shleifer et al., 2022), which places additional layer normalisations after
the attention output to improve training stability.

This encoder is an alternative backbone to ``DiTEncoder``.  The two share the
same ``MultiheadAttention`` building block but differ in how they condition on
global context: this encoder passes the context only to the feed-forward
sublayer via ``Dense``, whereas ``DiTEncoder`` uses adaLN modulation throughout.

References:
  Normformer: https://arxiv.org/abs/2110.09456
"""

from typing import Optional

import torch.nn as nn
from torch import BoolTensor, Tensor

from models.attention import MultiheadAttention
from models.dense import Dense


class TransformerEncoderLayer(nn.Module):
    """Single GPT-2 + Normformer encoder layer.

    Structure:
      1. LayerNorm → MultiHeadSelfAttention → LayerNorm → residual add
      2. (Optional) Dense FFN with context conditioning → residual add

    Args:
        embed_dim:      Token embedding dimension.
        mha_config:     Keyword arguments forwarded to ``MultiheadAttention``.
        dense_config:   Optional keyword arguments for the context-conditioned
                        ``Dense`` FFN.  ``None`` omits the FFN sublayer.
        context_dim:    Context dimension forwarded to the Dense FFN.
        edge_embed_dim: Edge feature dimension.  ``0`` disables edge features.
        update_edges:   Whether to update edge features via attention scores.
    """

    def __init__(
        self,
        embed_dim: int,
        mha_config: dict,
        dense_config: Optional[dict] = None,
        context_dim: int = 0,
        edge_embed_dim: int = 0,
        update_edges: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.edge_embed_dim = edge_embed_dim
        self.update_edges = update_edges

        self.mha = MultiheadAttention(
            embed_dim,
            edge_embed_dim=edge_embed_dim,
            update_edges=update_edges,
            **mha_config,
        )
        self.dense = (
            Dense(input_size=embed_dim, output_size=embed_dim, **dense_config)
            if dense_config
            else None
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        if edge_embed_dim > 0:
            self.enorm1 = nn.LayerNorm(edge_embed_dim)
            if update_edges:
                self.enorm2 = nn.LayerNorm(edge_embed_dim)

    def forward(
        self,
        x: Tensor,
        edge_x: Optional[Tensor] = None,
        mask: Optional[BoolTensor] = None,
        context: Optional[Tensor] = None,
        attn_mask: Optional[BoolTensor] = None,
        attn_bias: Optional[Tensor] = None,
    ) -> Tensor:
        """Encoder layer forward pass.

        Args:
            x:         (B, N, embed_dim) token features.
            edge_x:    (B, N, N, edge_embed_dim) optional edge features.
            mask:      (B, N) bool padding mask.  ``True`` = padded.
            context:   (B, context_dim) optional global conditioning for FFN.
            attn_mask: (B, N, N) optional explicit attention mask.
            attn_bias: (B, N, N, H) optional attention bias.

        Returns:
            Updated token features (B, N, embed_dim), or a tuple
            ``(tokens, edges)`` when *edge_x* is provided.
        """
        if edge_x is not None:
            xi, edge_xi = self.mha(
                self.norm1(x),
                edges=self.enorm1(edge_x),
                q_mask=mask,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
            )
        else:
            xi = self.mha(
                self.norm1(x),
                q_mask=mask,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
            )

        x = x + self.norm2(xi)

        if self.update_edges and edge_x is not None:
            edge_x = edge_x + self.enorm2(edge_xi)

        if self.dense is not None:
            x = x + self.dense(x, context)

        return (x, edge_x) if edge_x is not None else x


class TransformerEncoder(nn.Module):
    """Stack of Normformer encoder layers with optional output projection.

    Args:
        embed_dim:      Token embedding dimension.
        num_layers:     Number of stacked encoder layers.
        mha_config:     Forwarded to each ``TransformerEncoderLayer``.
        dense_config:   Forwarded to each layer's FFN.  ``None`` omits FFN.
        context_dim:    Global context dimension for the FFN.
        out_dim:        If > 0, a final linear layer projects to this dimension.
        edge_embed_dim: Edge feature dimension.  ``0`` disables edges.
        update_edges:   Whether to update edge features in each layer.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        mha_config: dict,
        dense_config: Optional[dict] = None,
        context_dim: int = 0,
        out_dim: int = 0,
        edge_embed_dim: int = 0,
        update_edges: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.out_dim = out_dim

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim, mha_config, dense_config, context_dim,
                edge_embed_dim,
                # Only the last layer never updates edges (no consumer after it)
                update_edges if i < num_layers - 1 else False,
            )
            for i in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.final_linear = nn.Linear(embed_dim, out_dim) if out_dim else None

    def forward(self, q: Tensor, edge_x: Optional[Tensor] = None, **kwargs) -> Tensor:
        """Pass tokens through all encoder layers, then normalise.

        Args:
            q:       (B, N, embed_dim) input token features.  The parameter is
                     named *q* for API compatibility with ``DiTEncoder``.
            edge_x:  Optional (B, N, N, edge_embed_dim) edge features.
            **kwargs: Forwarded to each layer (``mask``, ``context``, etc.).

        Returns:
            (B, N, embed_dim) updated tokens, or (B, N, out_dim) when
            *out_dim* > 0.
        """
        x = q
        for layer in self.layers:
            if edge_x is not None:
                x, edge_x = layer(x, edge_x, **kwargs)
            else:
                x = layer(x, **kwargs)
        x = self.final_norm(x)
        if self.final_linear is not None:
            x = self.final_linear(x)
        return x


class TransformerCrossAttentionLayer(TransformerEncoderLayer):
    """Transformer layer that attends from a query sequence to a key-value sequence.

    Subclasses ``TransformerEncoderLayer`` and adds a separate normalisation
    for the key-value sequence.

    Args:
        Same as ``TransformerEncoderLayer`` (edge args are unused).
    """

    def __init__(
        self,
        embed_dim: int,
        mha_config: dict,
        dense_config: Optional[dict] = None,
        context_dim: int = 0,
    ):
        super().__init__(embed_dim, mha_config, dense_config, context_dim)
        self.norm0 = nn.LayerNorm(embed_dim)

    def forward(  # type: ignore[override]
        self,
        query: Tensor,
        key_value: Tensor,
        query_mask: Optional[BoolTensor] = None,
        key_value_mask: Optional[BoolTensor] = None,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        """Cross-attention forward pass.

        Args:
            query:         (B, Lq, embed_dim) query tokens.
            key_value:     (B, Lk, embed_dim) key/value tokens.
            query_mask:    (B, Lq) padding mask for queries.
            key_value_mask: (B, Lk) padding mask for key/values.
            context:       (B, context_dim) optional conditioning for FFN.

        Returns:
            (B, Lq, embed_dim) updated query features.
        """
        query = query + self.norm2(
            self.mha(
                self.norm1(query),
                self.norm0(key_value),
                q_mask=query_mask,
                kv_mask=key_value_mask,
            )
        )
        if self.dense is not None:
            query = query + self.dense(query, context)
        return query
