"""Pixel value transformation for normalising image data.

Provides ``PixelTransformation``, a stateful transform that maps raw detector
pixel values into a standardised space suitable for flow matching, and back.

The chosen transform is:

    forward:  y = (log1p(x) - μ) / σ
    inverse:  x = clamp(expm1(y · σ + μ), min=0)

``log1p`` compresses the dynamic range of the sparse, heavy-tailed pixel
distribution while mapping zero pixels cleanly to zero.  Standardisation with
the training-set mean and standard deviation centres the model targets.

Typical usage::

    # Compute stats from training images and save to config
    transform = PixelTransformation.from_images(train_images)
    print(transform.mean, transform.std)   # copy into configs/model.yml

    # Apply at dataset load time
    transform = PixelTransformation.from_config(config['pixel_transform'])
    x_t = transform.forward(raw_pixels)
    x_raw = transform.inverse(model_output)
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import Tensor


class PixelTransformation:
    """Invertible log1p + standardisation transform for image pixel values.

    Args:
        mean: Mean of ``log1p(pixel)`` computed over the training set.
        std:  Standard deviation of ``log1p(pixel)`` over the training set.

    Note:
        Stats should be computed on the **training set only** (after
        cropping / resizing) using :meth:`from_images` or
        :meth:`compute_stats`, then stored in the model config so that
        validation and inference use the same normalisation.
    """

    def __init__(self, mean: float, std: float):
        if std <= 0:
            raise ValueError(f"std must be positive, got {std}")
        self.mean = float(mean)
        self.std = float(std)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict) -> PixelTransformation:
        """Instantiate from a config dict with keys ``'mean'`` and ``'std'``.

        Args:
            config: Dict containing at least ``{'mean': float, 'std': float}``.

        Returns:
            Configured ``PixelTransformation`` instance.
        """
        return cls(mean=config['mean'], std=config['std'])

    @classmethod
    def from_images(cls, images: Iterable[np.ndarray]) -> PixelTransformation:
        """Fit a transform to an iterable of raw pixel arrays.

        Computes ``log1p`` mean and standard deviation over all pixels across
        all images.  Call this on the **training set** to obtain the stats that
        go into ``configs/model.yml``.

        Args:
            images: Iterable of 2-D (or N-D) float32 numpy arrays with raw
                    pixel values.  All pixels (including zeros) are used.

        Returns:
            Fitted ``PixelTransformation``.

        Example::

            import h5py, numpy as np
            with h5py.File('data/train.h5', 'r') as f:
                imgs = f['images'][()]
            transform = PixelTransformation.from_images(imgs)
            print(f"mean={transform.mean:.4f}, std={transform.std:.4f}")
        """
        log_vals = np.concatenate([np.log1p(img.ravel()) for img in images])
        return cls(mean=float(log_vals.mean()), std=float(log_vals.std()))

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Map raw pixel values to normalised log space.

        Args:
            x: Pixel tensor with non-negative values (any shape).

        Returns:
            Transformed tensor of the same shape.
        """
        return (torch.log1p(x) - self.mean) / self.std

    def inverse(self, y: Tensor) -> Tensor:
        """Map normalised values back to raw pixel space.

        Args:
            y: Transformed tensor (any shape, matching a prior ``forward`` call).

        Returns:
            Reconstructed pixel tensor clamped to ``[0, ∞)``.
        """
        return torch.expm1(y * self.std + self.mean).clamp(min=0.0)

    def __repr__(self) -> str:
        return f"PixelTransformation(mean={self.mean:.4f}, std={self.std:.4f})"
