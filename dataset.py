"""CYGNO image dataset for flow matching.

Loads paired source / target images from an HDF5 file, crops each image to the
active (non-zero) pixel region, resizes the crop to a fixed square grid, and
returns a flat sequence of pixel values with corresponding 2-D grid coordinates.

The flat-sequence format is what ``FlowModel`` expects: each pixel is treated
as an independent token, and the model attends globally across all tokens.

Typical usage::

    from utility.transformation import PixelTransformation
    from dataset import CygnoDataset, collate_fn
    from torch.utils.data import DataLoader

    transform = PixelTransformation.from_config(config['pixel_transform'])
    ds = CygnoDataset('data/train.h5', pixel_transform=transform)
    loader = DataLoader(ds, batch_size=8, collate_fn=collate_fn)

    for batch in loader:
        # batch keys: 'pos', 'source', 'target', 'source_raw', 'target_raw'
        ...

When source and target images live under different HDF5 keys, pass them
explicitly::

    ds = CygnoDataset(
        'data/train.h5',
        source_key='images_distorted',
        target_key='images_clean',
        pixel_transform=transform,
    )
"""

from __future__ import annotations

from typing import Optional

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from utility.transformation import PixelTransformation


class CygnoDataset(Dataset):
    """HDF5 image dataset that exposes images as flat pixel sequences.

    Each ``__getitem__`` call:

    1. Loads source and target images from the HDF5 file.
    2. Computes the bounding box of active (> 0) pixels in the source image.
    3. Expands the box by *margin* pixels on each side.
    4. Applies the same crop to both source and target.
    5. Resizes the crop to *crop_size* × *crop_size* via nearest-neighbour
       sampling (no external dependencies).
    6. Flattens pixels to a 1-D sequence of length N = crop_size².
    7. Pairs each pixel with its normalised (x, y) coordinate in [-1, 1].
    8. Optionally applies a ``PixelTransformation`` to the pixel values.

    The returned dict contains both raw and transformed pixel values so that
    validation metrics can be computed in the original pixel space.

    Args:
        h5_path:         Path to the HDF5 file.
        source_key:      Dataset key for the source (conditioning) images.
                         Defaults to ``'images'``.
        target_key:      Dataset key for the target images.  Defaults to
                         ``'images'`` (same as source, for pre-training /
                         denoising experiments).
        crop_size:       Side length of the square output crop in pixels.
                         Defaults to 64.
        margin:          Extra pixels added around the non-zero bounding box
                         on each side.  Defaults to 16.
        pixel_transform: Optional ``PixelTransformation`` applied to both
                         source and target pixel values.  Raw values are
                         always returned alongside for evaluation.
        indices:         Optional list of integer sample indices to use.
                         Enables train / validation splitting without copying
                         data.  ``None`` uses all samples.
    """

    def __init__(
        self,
        h5_path: str,
        source_key: str = 'images',
        target_key: str = 'images',
        crop_size: int = 64,
        margin: int = 16,
        pixel_transform: Optional[PixelTransformation] = None,
        indices: Optional[list[int]] = None,
    ):
        self.h5_path = h5_path
        self.source_key = source_key
        self.target_key = target_key
        self.crop_size = crop_size
        self.margin = margin
        self.pixel_transform = pixel_transform

        with h5py.File(h5_path, 'r') as f:
            ds = f[source_key]
            if not isinstance(ds, h5py.Dataset):
                raise ValueError(f"'{source_key}' is not an HDF5 Dataset in {h5_path}")
            n_total = ds.shape[0]

        self.indices: list[int] = list(range(n_total)) if indices is None else list(indices)

        # Pixel coordinate grid is the same for every sample — compute once
        self._pos: Tensor = self._make_pixel_coords(crop_size)  # (N, 2)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_pixel_coords(crop_size: int) -> Tensor:
        """Return a (crop_size², 2) tensor of (x, y) coordinates in [-1, 1].

        Coordinates are arranged in row-major (C) order matching the flattened
        pixel array so that ``pos[i]`` corresponds to ``pixels[i]``.

        Args:
            crop_size: Number of pixels along each side of the square grid.

        Returns:
            Float32 tensor of shape (crop_size², 2).
        """
        lin = torch.linspace(-1.0, 1.0, crop_size)
        yy, xx = torch.meshgrid(lin, lin, indexing='ij')
        return torch.stack([xx.ravel(), yy.ravel()], dim=-1)

    @staticmethod
    def _active_bbox(img: np.ndarray, margin: int) -> tuple[int, int, int, int]:
        """Find the bounding box of non-zero pixels and expand it by *margin*.

        Args:
            img:    2-D float array of shape (H, W).
            margin: Number of pixels to add on each side of the tight box.

        Returns:
            ``(r0, r1, c0, c1)`` inclusive/exclusive row and column bounds,
            clipped to the image boundary.  Returns the full image extent when
            no non-zero pixel is found.
        """
        H, W = img.shape
        nz = np.argwhere(img > 0)
        if len(nz) == 0:
            return 0, H, 0, W
        r0 = max(0, int(nz[:, 0].min()) - margin)
        r1 = min(H, int(nz[:, 0].max()) + 1 + margin)
        c0 = max(0, int(nz[:, 1].min()) - margin)
        c1 = min(W, int(nz[:, 1].max()) + 1 + margin)
        return r0, r1, c0, c1

    @staticmethod
    def _resize(img: np.ndarray, size: int) -> np.ndarray:
        """Resize a 2-D array to *size* × *size* by nearest-neighbour sampling.

        Uses integer index selection rather than scipy / PIL so there are no
        additional dependencies.

        Args:
            img:  2-D float array of shape (H, W).
            size: Target side length.

        Returns:
            Float array of shape (size, size).
        """
        row_idx = np.linspace(0, img.shape[0] - 1, size).astype(int)
        col_idx = np.linspace(0, img.shape[1] - 1, size).astype(int)
        return img[np.ix_(row_idx, col_idx)]

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Load, crop, resize, and return one image pair.

        Args:
            idx: Index into ``self.indices`` (not the raw HDF5 index).

        Returns:
            Dict with the following entries — all float32 tensors:

            ``'pos'``        (N, 2)  normalised pixel coordinates
            ``'source'``     (N, 1)  transformed source pixel values
            ``'target'``     (N, 1)  transformed target pixel values
            ``'source_raw'`` (N, 1)  raw source pixel values
            ``'target_raw'`` (N, 1)  raw target pixel values

            where N = crop_size².
        """
        file_idx = self.indices[idx]

        with h5py.File(self.h5_path, 'r') as f:
            src_ds = f[self.source_key]
            tgt_ds = f[self.target_key]
            assert isinstance(src_ds, h5py.Dataset) and isinstance(tgt_ds, h5py.Dataset)
            source_img: np.ndarray = np.array(src_ds[file_idx], dtype=np.float32)
            target_img: np.ndarray = np.array(tgt_ds[file_idx], dtype=np.float32)

        # Determine crop from the source image and apply to both
        r0, r1, c0, c1 = self._active_bbox(source_img, self.margin)
        source_crop = self._resize(source_img[r0:r1, c0:c1], self.crop_size)
        target_crop = self._resize(target_img[r0:r1, c0:c1], self.crop_size)

        # Flatten to (N, 1) and convert to tensors
        source_raw = torch.from_numpy(source_crop.ravel()).unsqueeze(-1)  # (N, 1)
        target_raw = torch.from_numpy(target_crop.ravel()).unsqueeze(-1)  # (N, 1)

        if self.pixel_transform is not None:
            source = self.pixel_transform.forward(source_raw)
            target = self.pixel_transform.forward(target_raw)
        else:
            source = source_raw.clone()
            target = target_raw.clone()

        return {
            'pos':        self._pos.clone(),  # (N, 2)
            'source':     source,             # (N, 1) transformed
            'target':     target,             # (N, 1) transformed
            'source_raw': source_raw,         # (N, 1) raw
            'target_raw': target_raw,         # (N, 1) raw
        }


def collate_fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate a list of single-sample dicts into a batched dict.

    Because every sample has identical tensor shapes (fixed *crop_size*),
    collation is a simple stack — no padding is required.

    Args:
        batch: List of dicts as returned by ``CygnoDataset.__getitem__``.

    Returns:
        Dict mapping each key to a stacked tensor with a leading batch
        dimension B.
    """
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
