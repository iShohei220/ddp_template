# Trainer Template

Small Hugging Face `Trainer` template for starting new repositories.

It stays intentionally minimal, but already includes:

- `HfArgumentParser`-based argument handling
- standard `TrainingArguments` / `Trainer` integration
- Hugging Face `datasets` for data loading
- single-process and distributed training
- checkpoint save/resume
- `torch_compile` support through `TrainingArguments`
- basic dataset and model validation

## Layout

- `train.py`: `Trainer` entrypoint
- `arguments.py`: argument dataclasses and parsing helpers
- `configs/cifar10.json`: example argument file
- `dataset.py`: preprocessing and validation helpers
- `model.py`: model definition and model factory

## Running

CLI flags:

```bash
python train.py \
  --output_dir runs/ddp_template \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --num_train_epochs 3 \
  --learning_rate 1e-3 \
  --eval_strategy epoch \
  --save_strategy steps \
  --save_steps 100 \
  --train_sampling_strategy random \
  --remove_unused_columns false \
  --dataloader_drop_last true
```

Argument file:

```bash
python train.py configs/cifar10.json
```

Distributed on GPU:

```bash
torchrun --nproc_per_node=4 train.py configs/cifar10.json
```

CPU-only distributed runs should either use CLI flags or set `"use_cpu": true` in the JSON file before launching with `torchrun`.

## Template Conventions

- `TrainingArguments` is the main user-facing training surface
- `train.py` expects model `forward(pixel_values, labels=None)`
- `model.py` exposes `build_model(args)`
- `configs/cifar10.json` follows the flat argument style used by Hugging Face example scripts
- `remove_unused_columns=false` is required because the image pipeline uses lazy dataset transforms

## Current Example

The default example is a small CIFAR-10 classifier loaded through Hugging Face `datasets`:

- dataset: `cifar10`
- model: simple `ConvClassifier`
- task: classification with cross-entropy loss

## Notes For New Repos

- Use CLI flags or a flat JSON/YAML file instead of a nested custom config schema
- Keep custom arguments small and let `TrainingArguments` own the training loop settings
- If a dataset is large, consider setting `--cache_dir` to a path outside the repository
