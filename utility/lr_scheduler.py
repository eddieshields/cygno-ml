"""Learning-rate scheduler with warm-up and cosine annealing.

Provides ``CustomLRScheduler``, a three-phase schedule designed for
flow-matching and diffusion model training:

  Phase 1 — Warm-up:
      LR ramps from ``eta_min`` to ``base_lr`` over ``warm_start_epochs``
      epochs using a half-cosine curve.

  Phase 2 — Cosine annealing:
      LR decays from ``base_lr`` back to ``eta_min`` over ``cosine_epochs``
      epochs using a half-cosine curve.

  Phase 3 — Flat minimum:
      LR stays at ``eta_min`` for all remaining epochs.

Both epoch counts accept fractional values in (0, 1) as fractions of the
total number of training epochs (requires ``max_epoch`` to be set).
"""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """Warm-up + cosine annealing + flat floor learning-rate schedule.

    Args:
        optimizer:         Wrapped PyTorch optimiser.
        warm_start_epochs: Duration of the warm-up phase.  Integers are epoch
                           counts; floats in (0, 1) are fractions of
                           *max_epoch* and require *max_epoch* to be set.
        cosine_epochs:     Duration of the cosine annealing phase.  Same
                           fractional convention as *warm_start_epochs*.
        eta_min:           Minimum learning rate used as the floor and the
                           starting / ending value of the warm-up and
                           annealing phases.  Defaults to ``0.0``.
        last_epoch:        Index of the last completed epoch.  Pass ``-1``
                           (default) when starting fresh.
        max_epoch:         Total number of training epochs.  Required when
                           either phase duration is given as a fraction.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warm_start_epochs: float,
        cosine_epochs: float,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        max_epoch: int = None,
    ):
        if 0 < warm_start_epochs < 1:
            if max_epoch is None:
                raise ValueError("max_epoch is required when warm_start_epochs is fractional")
            warm_start_epochs = int(warm_start_epochs * max_epoch)

        if 0 < cosine_epochs < 1:
            if max_epoch is None:
                raise ValueError("max_epoch is required when cosine_epochs is fractional")
            cosine_epochs = int(cosine_epochs * max_epoch)

        self.warm_start_epochs = int(warm_start_epochs)
        self.cosine_epochs = int(cosine_epochs)
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute per-parameter-group learning rates for the current epoch.

        Returns:
            List of learning rates, one per parameter group.
        """
        e = self.last_epoch

        if e < self.warm_start_epochs:
            # Cosine warm-up: eta_min → base_lr
            progress = e / self.warm_start_epochs
            factor = (1 - math.cos(math.pi * progress)) / 2
        elif e < self.warm_start_epochs + self.cosine_epochs:
            # Cosine decay: base_lr → eta_min
            progress = (e - self.warm_start_epochs) / self.cosine_epochs
            factor = (1 + math.cos(math.pi * progress)) / 2
        else:
            # Flat floor
            return [self.eta_min for _ in self.base_lrs]

        return [self.eta_min + (base - self.eta_min) * factor for base in self.base_lrs]
