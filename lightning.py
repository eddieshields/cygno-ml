"""PyTorch Lightning training module for the CYGNO flow matching model.

``CygnoLightning`` wraps ``FlowModel`` in a ``pl.LightningModule`` and owns:

- HDF5 data loading via ``CygnoDataset`` (train + val splits)
- Training step with flow-matching loss logging
- Validation step with MSE in both transformed and raw pixel space
- Reconstruction image grids saved to disk at the end of each validation epoch
- AdamW + warm-up cosine annealing learning-rate schedule

Typical usage::

    from lightning import CygnoLightning
    import pytorch_lightning as pl, yaml

    with open('configs/model.yml') as f:
        cfg_model = yaml.safe_load(f)
    with open('configs/train.yml') as f:
        cfg_train = yaml.safe_load(f)

    model   = CygnoLightning(cfg_model, cfg_train)
    trainer = pl.Trainer(max_epochs=cfg_train['num_epochs'], accelerator='gpu')
    trainer.fit(model)

See :mod:`train` for the command-line entry point.
"""

from __future__ import annotations

import os

import h5py
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import Tensor
from torch.utils.data import DataLoader

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)

from dataset import CygnoDataset, collate_fn
from models.flow_model import FlowModel
from utility.lr_scheduler import CustomLRScheduler
from utility.transformation import PixelTransformation


class CygnoLightning(pl.LightningModule):
    """Lightning module for image-to-image flow matching on CYGNO data.

    Args:
        config_model: Model configuration dict loaded from ``configs/model.yml``.
                      Must contain a ``'flow_model'`` key (forwarded to
                      ``FlowModel``) and a ``'pixel_transform'`` key
                      (forwarded to ``PixelTransformation.from_config``).
        config_train: Training configuration dict loaded from
                      ``configs/train.yml``.  See :meth:`_make_dataset` and
                      :meth:`configure_optimizers` for the expected keys.
    """

    def __init__(self, config_model: dict, config_train: dict):
        super().__init__()
        self.save_hyperparameters()

        self.config_model = config_model
        self.config_train = config_train

        self.net = FlowModel(config_model['flow_model'])
        self.pixel_transform = PixelTransformation.from_config(config_model['pixel_transform'])

        # Accumulated per-step validation outputs, cleared at epoch end
        self._val_outputs: list[dict] = []

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _make_dataset(self, split: str) -> CygnoDataset:
        """Build a ``CygnoDataset`` for the given split.

        Reads the following keys from ``config_train``:

        - ``'{split}_path'``   (str)  — path to the HDF5 file
        - ``'source_key'``     (str, default ``'images'``)
        - ``'target_key'``     (str, default ``'images'``)
        - ``'crop_size'``      (int, default ``64``)
        - ``'margin'``         (int, default ``16``)
        - ``'limit_{split}'``  (int, default ``-1``) — cap on number of samples;
          ``-1`` means no cap.

        Args:
            split: Either ``'train'`` or ``'val'``.

        Returns:
            Configured ``CygnoDataset`` instance.
        """
        cfg = self.config_train
        h5_path    = cfg[f'{split}_path']
        source_key = cfg.get('source_key', 'images')
        target_key = cfg.get('target_key', 'images')

        with h5py.File(h5_path, 'r') as f:
            ds = f[source_key]
            if not isinstance(ds, h5py.Dataset):
                raise ValueError(f"'{source_key}' is not an HDF5 Dataset in {h5_path}")
            n_total = ds.shape[0]

        limit   = cfg.get(f'limit_{split}', -1)
        indices = list(range(n_total if limit == -1 else min(limit, n_total)))

        return CygnoDataset(
            h5_path=h5_path,
            source_key=source_key,
            target_key=target_key,
            crop_size=cfg.get('crop_size', 64),
            margin=cfg.get('margin', 16),
            pixel_transform=self.pixel_transform,
            indices=indices,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the shuffled training ``DataLoader``."""
        ds = self._make_dataset('train')
        return DataLoader(
            ds,
            batch_size=self.config_train['batch_size_train'],
            shuffle=True,
            num_workers=self.config_train.get('num_workers', 2),
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the deterministic validation ``DataLoader``."""
        ds = self._make_dataset('val')
        return DataLoader(
            ds,
            batch_size=self.config_train['batch_size_val'],
            shuffle=False,
            num_workers=self.config_train.get('num_workers', 2),
            collate_fn=collate_fn,
            pin_memory=True,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        """Compute flow-matching loss and log diagnostics.

        Args:
            batch:     Batch dict from ``CygnoDataset`` (keys: pos, source,
                       target, source_raw, target_raw).
            batch_idx: Index of this batch within the epoch (unused).

        Returns:
            Scalar loss tensor for back-propagation.
        """
        loss, stats, _ = self.net.get_loss(batch)
        B = batch['source'].shape[0]
        self.log('train/loss', loss, batch_size=B, prog_bar=True)
        self.log_dict({f'train/{k}': v for k, v in stats.items()})
        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Generate samples and record per-batch MSE in transformed + raw space.

        Results are buffered in ``_val_outputs`` for aggregation at epoch end.
        The first batch's data is also saved for image-grid visualisation.

        Args:
            batch:     Batch dict from ``CygnoDataset``.
            batch_idx: Index of this batch within the validation epoch.
        """
        with torch.no_grad():
            pred = self.net.generate_samples(batch)  # (B, N, 1) — transformed space

        target     = batch['target']      # (B, N, 1) transformed
        target_raw = batch['target_raw']  # (B, N, 1) raw

        loss_t   = F.mse_loss(pred, target)
        pred_raw = self.pixel_transform.inverse(pred)
        loss_raw = F.mse_loss(pred_raw, target_raw)

        self._val_outputs.append({
            'loss_t':   loss_t.item(),
            'loss_raw': loss_raw.item(),
            # Only keep the first batch for visualisation to avoid OOM
            'pred':  pred.detach().cpu() if batch_idx == 0 else None,
            'batch': {k: v.cpu() for k, v in batch.items()} if batch_idx == 0 else None,
        })

    def on_validation_epoch_end(self) -> None:
        """Aggregate buffered validation outputs, log epoch metrics, and save images."""
        outputs = self._val_outputs
        val_loss_t   = float(np.mean([o['loss_t']   for o in outputs]))
        val_loss_raw = float(np.mean([o['loss_raw'] for o in outputs]))

        self.log('val/loss',     val_loss_t,   prog_bar=True)
        self.log('val/loss_raw', val_loss_raw, prog_bar=True)

        first = next((o for o in outputs if o['pred'] is not None), None)
        if first is not None:
            n_display = self.config_train.get('n_image_displays', 3)
            self._log_image_grid(first['batch'], first['pred'], n_display)

        self._val_outputs.clear()

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _log_image_grid(
        self,
        batch: dict,
        pred: Tensor,
        n_display: int,
    ) -> None:
        """Save a grid of source / target / predicted / diff images to disk.

        Creates one PNG per displayed sample under ``config_train['plot_dir']``
        (default: ``'plots'``).  Filenames encode the epoch and sample index so
        that images from different epochs do not overwrite each other.

        Args:
            batch:     CPU batch dict containing ``'source_raw'`` and
                       ``'target_raw'`` tensors of shape (B, N, 1).
            pred:      CPU tensor of shape (B, N, 1) — model output in
                       transformed pixel space.
            n_display: Maximum number of samples to visualise.
        """
        crop_size = self.config_train.get('crop_size', 64)
        pred_raw  = self.pixel_transform.inverse(pred)

        save_dir = self.config_train.get('plot_dir', 'plots')
        os.makedirs(save_dir, exist_ok=True)

        for i in range(min(n_display, pred.shape[0])):
            source_img = batch['source_raw'][i].reshape(crop_size, crop_size).numpy()
            target_img = batch['target_raw'][i].reshape(crop_size, crop_size).numpy()
            pred_img   = pred_raw[i].reshape(crop_size, crop_size).numpy()
            diff_img   = pred_img - target_img

            fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
            vmax = max(source_img.max(), target_img.max(), pred_img.max(), 1e-3)

            panels = [
                ('Source',          source_img, 'viridis',  0,         vmax),
                ('Target',          target_img, 'viridis',  0,         vmax),
                ('Predicted',       pred_img,   'viridis',  0,         vmax),
                ('Pred - Target',   diff_img,   'RdBu_r',  -vmax / 2,  vmax / 2),
            ]
            for ax, (title, img, cmap, vmin, vmax_panel) in zip(axes, panels):
                im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax_panel)
                ax.set_title(title)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046)

            fig.suptitle(f'Epoch {self.current_epoch} — sample {i}')
            plt.tight_layout()
            fig.savefig(
                f'{save_dir}/epoch{self.current_epoch:03d}_sample{i}.png',
                dpi=100,
            )
            plt.close(fig)

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict:
        """Set up AdamW with an optional warm-up cosine annealing schedule.

        Reads the following keys from ``config_train``:

        - ``'learning_rate'``  (float) — peak AdamW learning rate
        - ``'lr_scheduler'``   (dict, optional) — if absent, constant LR is used.
          Sub-keys:

          - ``'warm_start_epochs'`` (float)
          - ``'cosine_epochs'``     (float)
          - ``'eta_min'``           (float)
          - ``'last_epoch'``        (int, default ``-1``)
          - ``'max_epochs'``        (int | ``'take_as_num_epochs'``)

        Returns:
            Lightning-compatible dict with ``'optimizer'`` and optionally
            ``'lr_scheduler'``.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config_train['learning_rate'],
        )

        lr_cfg = self.config_train.get('lr_scheduler')
        if lr_cfg is None:
            return {'optimizer': optimizer}

        max_epoch = (
            self.config_train['num_epochs']
            if lr_cfg.get('max_epochs') == 'take_as_num_epochs'
            else lr_cfg.get('max_epochs')
        )
        scheduler = CustomLRScheduler(
            optimizer,
            warm_start_epochs=lr_cfg['warm_start_epochs'],
            cosine_epochs=lr_cfg['cosine_epochs'],
            eta_min=lr_cfg['eta_min'],
            last_epoch=lr_cfg.get('last_epoch', -1),
            max_epoch=max_epoch,
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def on_train_epoch_end(self) -> None:
        """Log the current learning rate and advance the scheduler by one epoch."""
        self.log('train/lr', self.optimizers().param_groups[0]['lr'])
        if self.config_train.get('lr_scheduler') is not None:
            self.lr_schedulers().step()
