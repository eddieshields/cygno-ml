"""Low-level utility functions and modules shared across all model files.

Provides:
  - Tensor shape helpers  (add_dims)
  - Attention helpers     (masked_softmax, merge_masks)
  - Context concatenation (attach_context)
  - AdaLN modulation      (modulate)
  - Sinusoidal time embedding (TimestepEmbedder)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import softmax


# ---------------------------------------------------------------------------
# Tensor shape helpers
# ---------------------------------------------------------------------------

def add_dims(x: Tensor, ndim: int) -> Tensor:
    """Expand a tensor to *ndim* dimensions by inserting size-1 dims after dim 0.

    Args:
        x:    Input tensor with shape (B, ...).
        ndim: Target number of dimensions.  Must be >= x.dim().

    Returns:
        Tensor with shape (B, 1, ..., 1, *x.shape[1:]) where extra 1s fill the gap.

    Raises:
        ValueError: If *ndim* < x.dim().
    """
    if (dim_diff := ndim - x.dim()) < 0:
        raise ValueError(
            f"Target ndim ({ndim}) must be >= input ndim ({x.dim()})"
        )
    if dim_diff > 0:
        x = x.view(x.shape[0], *([1] * dim_diff), *x.shape[1:])
    return x


# ---------------------------------------------------------------------------
# Attention helpers
# ---------------------------------------------------------------------------

def masked_softmax(x: Tensor, mask: Optional[Tensor], dim: int = -1) -> Tensor:
    """Softmax that sets attention weights to 0 for masked (padded) positions.

    Args:
        x:    Attention logits, any shape.
        mask: Boolean tensor broadcastable to x.  ``True`` = position is padded
              and should be excluded.  ``None`` means no masking.
        dim:  Dimension along which to apply softmax.

    Returns:
        Attention weights with the same shape as *x*.  Masked positions are 0.
    """
    if mask is not None:
        mask = add_dims(mask, x.dim())
        x = x.masked_fill(mask, -torch.inf)
    x = softmax(x, dim=dim)
    if mask is not None:
        x = x.masked_fill(mask, 0.0)
    return x


def merge_masks(
    q_mask: Optional[Tensor],
    kv_mask: Optional[Tensor],
    attn_mask: Optional[Tensor],
    q_shape: torch.Size,
    k_shape: torch.Size,
    device: torch.device,
) -> Optional[Tensor]:
    """Combine padding masks and an explicit attention mask into one mask.

    Follows the PyTorch transformer convention: ``True`` = padded / blocked.

    Args:
        q_mask:    (B, Lq) bool padding mask for queries.  ``None`` = all valid.
        kv_mask:   (B, Lk) bool padding mask for keys/values.  ``None`` = all valid.
        attn_mask: (B, Lq, Lk) bool explicit attention mask.  ``None`` = no mask.
        q_shape:   Shape of the query tensor.
        k_shape:   Shape of the key tensor.
        device:    Device for newly created tensors.

    Returns:
        Combined (B, Lq, Lk) bool mask, or ``None`` if all inputs are ``None``.
    """
    merged: Optional[Tensor] = None

    if q_mask is not None or kv_mask is not None:
        if q_mask is None:
            q_mask = torch.full(q_shape[:-1], False, device=device)
        if kv_mask is None:
            kv_mask = torch.full(k_shape[:-1], False, device=device)
        merged = q_mask.unsqueeze(-1) | kv_mask.unsqueeze(-2)

    if attn_mask is not None:
        merged = attn_mask if merged is None else (attn_mask | merged)

    return merged


# ---------------------------------------------------------------------------
# Context concatenation
# ---------------------------------------------------------------------------

def attach_context(x: Tensor, context: Tensor) -> Tensor:
    """Concatenate a lower-dimensional context tensor to a higher-dimensional input.

    Handles the common case of appending a global (batch-level) feature vector
    to a per-token feature matrix by broadcasting along the token dimension.

    Example::

        x       shape (B, N, F)   per-token features
        context shape (B, C)      global features
        output  shape (B, N, F+C) context repeated for each token

    Args:
        x:       Input tensor with shape (B, [L1, L2, ...], F).
        context: Context tensor with shape (B, [1, ...], C).  Must have fewer
                 or equal dimensions to *x*.

    Returns:
        Concatenation of *x* and the broadcast-expanded context along the last
        dimension.

    Raises:
        RuntimeError: If *context* is ``None``.
        ValueError:   If *context* has more dimensions than *x*.
    """
    if context is None:
        raise RuntimeError("attach_context: context tensor is None")
    if (dim_diff := x.dim() - context.dim()) < 0:
        raise ValueError(
            f"context has more dimensions ({context.dim()}) than input ({x.dim()})"
        )
    if dim_diff > 0:
        context = add_dims(context, x.dim()).expand(*x.shape[:-1], -1)
    return torch.cat([x, context], dim=-1)


# ---------------------------------------------------------------------------
# AdaLN modulation
# ---------------------------------------------------------------------------

def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Adaptive layer-norm modulation: ``x * (1 + scale) + shift``.

    Used inside DiT blocks to condition normalised activations on a global
    context vector.  *shift* and *scale* have shape (B, D); a singleton token
    dimension is inserted automatically so that the operation broadcasts over
    the full token sequence.

    Args:
        x:     (B, N, D) token features after layer normalisation.
        shift: (B, D)    additive shift from the adaLN MLP.
        scale: (B, D)    multiplicative scale from the adaLN MLP.

    Returns:
        Modulated tensor with the same shape as *x*.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """Embeds a scalar diffusion timestep t ∈ [0, 1] into a dense vector.

    Uses sinusoidal positional encoding (à la Transformer / DDPM) followed by
    a two-layer MLP to produce a fixed-size embedding.

    Args:
        hidden_size:             Output embedding dimension.
        frequency_embedding_size: Number of sinusoidal frequency features.
                                  Defaults to 256.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        """Sinusoidal embedding of scalar timesteps.

        Args:
            t:          (B,) float tensor of timesteps in [0, 1].
            dim:        Number of embedding dimensions.
            max_period: Controls the minimum sinusoidal frequency.

        Returns:
            (B, dim) embedding tensor.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: (B,) timestep values in [0, 1].

        Returns:
            (B, hidden_size) timestep embeddings.
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)
