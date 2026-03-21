# DDP Template

Small PyTorch + DDP training template for starting new repositories.

It is intentionally minimal, but already includes:

- typed config loading from YAML
- dataset/model separation
- single-process and distributed training
- checkpoint save/resume
- TensorBoard logging
- `torch.compile` support
- basic config and dataset validation

## Layout

- `train.py`: training loop and distributed setup
- `config.py`: typed config schema and YAML loader
- `conf/config.yaml`: default config
- `dataset.py`: dataset factory and preprocessing/download logic
- `model.py`: model definition and model factory

## Running

Single process:

```bash
python train.py --config_path conf/config.yaml
```

Distributed:

```bash
torchrun --nproc_per_node=4 train.py --config_path conf/config.yaml
```

## Template Conventions

- `dataset.*` holds data-side shape metadata
- `config.py` syncs model-side shape fields from dataset config when appropriate
- `train.py` expects model `forward(...)` to return the training loss
- `dataset.py` exposes `load_dataset(cfg, split=...)`
- `model.py` exposes `build_model(cfg)`

## Current Example

The default example is a small CIFAR-10 classifier:

- dataset: `cifar10`
- model: simple `ConvClassifier`
- task: classification with cross-entropy loss

## Notes For New Repos

- If a dataset is large, consider setting `dataset.root` to a path outside the repository
- Keep `conf/config.yaml` as the main user-editable surface
- Add project-specific modules only after the flat layout starts to feel crowded
