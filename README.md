# cygno-ml

`cygno-ml` trains a conditional flow matching model for CYGNO image-to-image
tasks. It reads sparse detector images from HDF5 files, crops each image around
the active pixel region, resizes the crop to a fixed square grid, and trains a
transformer-based flow model to generate target pixels conditioned on source
pixels and 2D pixel coordinates.

The current training stack is:

- `train.py`: command-line training entry point.
- `lightning.py`: PyTorch Lightning module, data loaders, validation, plots,
  optimizer, and scheduler.
- `dataset.py`: HDF5 dataset that returns flat pixel sequences plus normalized
  pixel coordinates.
- `models/flow_model.py`: conditional flow matching model with a DiT or
  GPT-2+Normformer style transformer backbone.
- `configs/model.yml`: model architecture and pixel normalization.
- `configs/train.yml`: data paths, batch sizes, learning rate, schedule, and
  output locations.

## Setup

Create and activate an environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install PyTorch first, choosing the wheel or conda package that matches your
machine. For example, for CPU-only development:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

For CUDA training, install a CUDA-enabled PyTorch build instead, then install
the remaining requirements:

```bash
pip install -r requirements.txt
```

If your Python executable is named `python` inside the environment, you can use
`python` in place of `python3` in the commands below.

## Data

Training and validation data are configured in `configs/train.yml`:

```yaml
train_path: data/small_dataset.h5
val_path: data/small_dataset.h5
source_key: images
target_key: images
```

The bundled `data/small_dataset.h5` contains an `images` dataset with shape
`(498, 512, 512)`. By default, source and target both use `images`, which is
useful for smoke tests and reconstruction-style experiments. For supervised
image-to-image training, set `source_key` and `target_key` to different HDF5
datasets with matching leading dimensions.

Each sample is processed as follows:

1. Load the source and target image.
2. Find the non-zero bounding box in the source image.
3. Expand the box by `margin` pixels.
4. Apply the same crop to source and target.
5. Resize to `crop_size x crop_size`.
6. Flatten pixels to a sequence of length `crop_size ** 2`.
7. Attach normalized `(x, y)` coordinates in `[-1, 1]`.
8. Apply the log1p pixel transform from `configs/model.yml`.

When moving to a new dataset, recompute the pixel transform statistics on the
training set and update:

```yaml
pixel_transform:
  mean: ...
  std: ...
```

These are the mean and standard deviation of `log1p(pixel)` values after the
same preprocessing used for training.

## Training

Run a GPU training job with the default configs:

```bash
python3 train.py \
  --config_model configs/model.yml \
  --config_train configs/train.yml \
  --gpu 0
```

Run on CPU:

```bash
python3 train.py -cm configs/model.yml -ct configs/train.yml -g cpu
```

Use lower matmul precision for faster TensorFloat-32 matmuls on supported
NVIDIA GPUs:

```bash
python3 train.py -cm configs/model.yml -ct configs/train.yml -g 0 -p medium
```

Resume from a Lightning checkpoint:

```bash
python3 train.py \
  -cm configs/model.yml \
  -ct configs/train.yml \
  -g 0 \
  -r outputs/lightning_logs/version_0/checkpoints/last.ckpt
```

Important training config fields:

- `num_epochs`: total epochs.
- `batch_size_train`, `batch_size_val`: train and validation batch sizes.
- `learning_rate`: AdamW learning rate.
- `limit_train`, `limit_val`: cap dataset size for quick runs; `-1` uses all
  available samples.
- `crop_size`, `margin`: image preprocessing controls.
- `eval_every_n_epoch`: validation frequency.
- `lr_scheduler`: warm-up, cosine decay, and minimum learning-rate settings.
- `output_dir`: Lightning logs and checkpoints.
- `plot_dir`: validation image grids.

## Outputs

Training writes into `output_dir`, which defaults to `outputs`.

Lightning checkpoints are saved every epoch, keeping the best three checkpoints
by `val/loss_raw` plus `last.ckpt`. Validation plots are saved as PNG files in
`plot_dir`, with panels for source, target, prediction, and prediction error.

The main logged metrics are:

- `train/loss`: flow matching loss in transformed pixel space.
- `train/lr`: current learning rate.
- `val/loss`: validation MSE in transformed pixel space.
- `val/loss_raw`: validation MSE after inverse pixel transformation.

## Model Notes

`FlowModel` treats each pixel as a token. Per-pixel conditioning combines:

- normalized 2D pixel position;
- transformed source pixel value;
- raw source pixel value.

During training, conditional flow matching samples a noisy point between
Gaussian noise and the transformed target image, and the model predicts the
velocity field. During validation, the model integrates the learned ODE from
noise to a generated target image using `torchdiffeq`.

The default architecture uses a DiT-style transformer with full self-attention
over the fixed-size pixel grid. You can switch the backbone with
`flow_model.transformer.type` in `configs/model.yml`.

## Quick Smoke Test

For a short run on the bundled dataset, reduce the limits and epochs in
`configs/train.yml`, for example:

```yaml
limit_train: 32
limit_val: 8
num_epochs: 2
batch_size_train: 4
batch_size_val: 4
```

Then run:

```bash
python3 train.py -cm configs/model.yml -ct configs/train.yml -g cpu
```

This should create an `outputs` directory with logs, checkpoints, and validation
plots.
