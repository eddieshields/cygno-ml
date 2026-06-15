"""Command-line entry point for training the CYGNO flow matching model.

Loads model and training configs from YAML files, builds a
``CygnoLightning`` module, and runs PyTorch Lightning training with:

- ``ModelCheckpoint``: saves the top-3 checkpoints by ``val/loss_raw`` plus
  the most recent (``last.ckpt``).
- ``LearningRateMonitor``: logs the learning rate every epoch.

Usage::

    python train.py \\
        --config_model configs/model.yml \\
        --config_train configs/train.yml \\
        --gpu 0

To resume from a saved checkpoint::

    python train.py -cm configs/model.yml -ct configs/train.yml -r outputs/last.ckpt
"""

from __future__ import annotations

import argparse
import os

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from lightning import CygnoLightning


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed :class:`argparse.Namespace` with attributes:
        ``config_model``, ``config_train``, ``gpu``, ``precision``,
        ``resume``.
    """
    p = argparse.ArgumentParser(
        description=(
            'Train the CYGNO image-to-image flow matching model.  '
            'Supply a model config YAML and a training config YAML; '
            'all other settings are optional.'
        ),
    )
    p.add_argument(
        '--config_model', '-cm',
        required=True,
        metavar='PATH',
        help='Path to the model configuration YAML (e.g. configs/model.yml).',
    )
    p.add_argument(
        '--config_train', '-ct',
        required=True,
        metavar='PATH',
        help='Path to the training configuration YAML (e.g. configs/train.yml).',
    )
    p.add_argument(
        '--gpu', '-g',
        default='0',
        metavar='INDEX',
        help=(
            'CUDA device index to use (e.g. "0", "1").  '
            'Pass "cpu" to force CPU training.'
        ),
    )
    p.add_argument(
        '--precision', '-p',
        default='highest',
        choices=['highest', 'high', 'medium'],
        help=(
            'torch.set_float32_matmul_precision level.  '
            '"medium" enables TF32 on Ampere+ GPUs for faster matmuls.'
        ),
    )
    p.add_argument(
        '--resume', '-r',
        default=None,
        metavar='CKPT_PATH',
        help='Path to a Lightning checkpoint to resume training from.',
    )
    return p.parse_args()


def main() -> None:
    """Load configs, build trainer, and launch training."""
    args = parse_args()

    if args.gpu != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        torch.set_float32_matmul_precision(args.precision)

    with open(args.config_model) as f:
        config_model = yaml.safe_load(f)
    with open(args.config_train) as f:
        config_train = yaml.safe_load(f)

    model = CygnoLightning(config_model, config_train)

    checkpoint_cb = ModelCheckpoint(
        monitor='val/loss_raw',
        mode='min',
        save_top_k=3,
        save_last=True,
        filename='{epoch}-{val/loss_raw:.4f}',
        every_n_epochs=1,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    accelerator = 'cpu' if args.gpu == 'cpu' else 'gpu'
    trainer = pl.Trainer(
        max_epochs=config_train['num_epochs'],
        accelerator=accelerator,
        devices=1,
        default_root_dir=config_train.get('output_dir', 'outputs'),
        callbacks=[checkpoint_cb, lr_monitor],
        check_val_every_n_epoch=config_train.get('eval_every_n_epoch', 1),
        log_every_n_steps=1,
    )

    trainer.fit(model, ckpt_path=args.resume)


if __name__ == '__main__':
    main()
