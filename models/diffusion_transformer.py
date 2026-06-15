"""Diffusion Transformer (DiT) encoder.

Implements the DiT architecture from Peebles & Xie (2023).  Each layer applies
adaptive layer normalisation (adaLN) conditioned on a global context vector
(containing the diffusion timestep and image-level conditioning features) to
modulate its self-attention and feed-forward sublayers.

Reference: https://arxiv.org/abs/2212.09748
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from models.attention import MultiheadAttention
from models.dense import Dense
from models.utils import modulate


class DiTLayer(nn.Module):
    """Single DiT transformer layer with adaLN conditioning.

    Applies context-adaptive layer normalisation (adaLN) to both the
    multi-head self-attention and the feed-forward sublayers.  The shift,
    scale, and gating parameters are produced by a small MLP applied to the
    global context vector.

    Args:
        embed_dim:   Token embedding dimension.
        context_dim: Dimension of the global conditioning context.
        mha_config:  Keyword arguments forwarded to ``MultiheadAttention``.
        dense_config: Optional keyword arguments for the feed-forward
                      ``Dense`` network.  ``None`` omits the FFN sublayer.
    """

    def __init__(
        self,
        embed_dim: int,
        context_dim: int,
        mha_config: dict,
        dense_config: Optional[dict] = None,
    ):
        super().__init__()

        self.mha = MultiheadAttention(embed_dim, **mha_config)
        self.dense = (
            Dense(input_size=embed_dim, output_size=embed_dim, **dense_config)
            if dense_config
            else None
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Single MLP produces 6 × embed_dim parameters:
        #   (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(context_dim, 6 * embed_dim, bias=True),
        )

    def forward(
        self,
        q: Tensor,
        q_mask: Optional[Tensor] = None,
        k: Optional[Tensor] = None,
        kv_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
    ) -> Tensor:
        """DiT layer forward pass.

        Args:
            q:         (B, N, embed_dim) input token features.
            q_mask:    (B, N) optional bool padding mask for queries.
            k:         (B, M, embed_dim) optional key/value tokens for
                       cross-attention.  ``None`` → self-attention.
            kv_mask:   (B, M) optional padding mask for *k*.
            context:   (B, context_dim) global conditioning vector.
            attn_mask: (B, N, M) optional explicit attention mask.
            attn_bias: (B, N, M, H) optional attention bias.

        Returns:
            (B, N, embed_dim) updated token features.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(context).chunk(6, dim=1)
        )

        if k is None:
            q_attn = self.mha(
                q=modulate(self.norm1(q), shift_msa, scale_msa),
                q_mask=q_mask,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
            )
        else:
            q_attn = self.mha(
                q=q,
                k=modulate(self.norm1(k), shift_msa, scale_msa),
                q_mask=q_mask,
                kv_mask=kv_mask,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
            )

        q = q + gate_msa.unsqueeze(1) * q_attn

        if self.dense is not None:
            q_mlp = self.dense(modulate(self.norm2(q), shift_mlp, scale_mlp), context)
            q = q + gate_mlp.unsqueeze(1) * q_mlp

        return q


class DiTEncoder(nn.Module):
    """Stack of DiT layers followed by a final layer normalisation.

    Args:
        embed_dim:   Token embedding dimension throughout the encoder.
        num_layers:  Number of stacked DiT layers.
        mha_config:  Forwarded to each ``DiTLayer``.
        dense_config: Forwarded to each ``DiTLayer`` for the FFN sublayer.
        context_dim: Dimension of the global conditioning context.
        out_dim:     If > 0, a final linear layer projects tokens to this
                     dimension.  Defaults to 0 (disabled).
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        mha_config: dict,
        dense_config: Optional[dict] = None,
        context_dim: int = 0,
        out_dim: int = 0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.out_dim = out_dim

        self.layers = nn.ModuleList([
            DiTLayer(embed_dim, context_dim, mha_config, dense_config)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.final_linear = nn.Linear(embed_dim, out_dim) if out_dim else None

    def forward(self, q: Tensor, **kwargs) -> Tensor:
        """Pass *q* through all DiT layers, then normalise.

        Args:
            q:       (B, N, embed_dim) input token features.
            **kwargs: Forwarded verbatim to each ``DiTLayer.forward`` (e.g.
                     ``context``, ``q_mask``, ``attn_mask``).

        Returns:
            (B, N, embed_dim) updated tokens, or (B, N, out_dim) when
            *out_dim* > 0.
        """
        for layer in self.layers:
            q = layer(q, **kwargs)
        q = self.final_norm(q)
        if self.final_linear is not None:
            q = self.final_linear(q)
        return q
