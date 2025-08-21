import argparse
import os
import random
import yaml

import numpy as np
from tqdm import tqdm
import torch
from torch.distributed import destroy_process_group, init_process_group, reduce
from torch import nn
from torch.nn.functional import cross_entropy
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler

from config import Config
from dataset import load_dataset


# Global Variables
CUDA_IS_AVAILABLE = torch.cuda.is_available()
RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])


def train(
    cfg, 
    model, 
    optimizer, 
    scheduler, 
    dataloader,
    writer, 
    epoch, 
):
    model.train()
    running_loss = 0.0
    for x, y in dataloader:
        optimizer.zero_grad()

        pred = model(x)
        loss = cross_entropy(pred, y.to(pred.device)).mean()
        loss.backward()

        optimizer.step()
        scheduler.step()

        running_loss += loss.detach()

    reduce(running_loss, 0)

    if RANK == 0:
        writer.add_scalar(
            f'loss/train',
            running_loss.item() / (WORLD_SIZE * cfg.train.steps_per_epoch),
            cfg.train.steps_per_epoch * epoch
        )


@torch.no_grad()
def test(
    cfg,
    model,
    dataloader,
    writer,
    epoch,
):
    model.eval()
    running_loss = 0.0
    for x, y in dataloader:
        pred = model(x)
        loss = cross_entropy(pred, y.to(pred.device)).mean()
        running_loss += loss.detach()

    reduce(running_loss, 0)

    writer.add_scalar(
        f'loss/test',
        running_loss.item() / (WORLD_SIZE * cfg.test.steps_per_epoch),
        cfg.test.steps_per_epoch * epoch
    )


def create_dataloader(cfg):
    train_set = load_dataset(cfg.dataset, train=True)
    test_set = load_dataset(cfg.dataset, train=False)

    kwargs = {'num_workers': cfg.num_workers, 'pin_memory': True, "persistent_workers": cfg.num_workers > 0}

    generator = torch.Generator()
    generator.manual_seed(WORLD_SIZE * cfg.seed + RANK)

    train_sampler = RandomSampler(
        train_set,
        replacement=True,
        num_samples=cfg.train.steps_per_epoch * cfg.train.batch_size,
        generator=generator
    )
    test_sampler = RandomSampler(
        test_set,
        replacement=True,
        num_samples=cfg.test.steps_per_epoch * cfg.test.batch_size,
        generator=generator
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.train.batch_size,
        sampler=train_sampler,
        **kwargs
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.test.batch_size,
        sampler=test_sampler,
        **kwargs
    )

    return train_loader, test_loader


if __name__ == "__main__":
    # DDP Setup
    init_process_group(backend="nccl" if CUDA_IS_AVAILABLE else "gloo")
    if CUDA_IS_AVAILABLE:
        torch.cuda.set_device(LOCAL_RANK)

    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Distributed Data Parallel Training"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "conf/config.yaml")
    )

    args = parser.parse_args()
    cfg = Config(**yaml.safe_load(open(args.config_path)))
    
    # Set seed
    local_seed = WORLD_SIZE * cfg.seed + RANK
    torch.manual_seed(local_seed)
    random.seed(local_seed)
    np.random.seed(local_seed)

    # Dataset
    train_loader, test_loader = create_dataloader(cfg)

    # Model
    model = nn.Sequential(
        nn.Conv2d(cfg.model.in_channels, cfg.model.out_channels, 1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    ).to(LOCAL_RANK if CUDA_IS_AVAILABLE else "cpu")
    model = DistributedDataParallel(model, device_ids=[LOCAL_RANK] if CUDA_IS_AVAILABLE else None)

    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=cfg.optim.betas,
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay
    )
    scheduler = LambdaLR(optimizer, lambda step: 1.0)

    # TensorBoard
    writer = SummaryWriter() if RANK == 0 else None

    for epoch in tqdm(range(cfg.train.num_epochs)):
        train(
            cfg, 
            model, 
            optimizer, 
            scheduler, 
            train_loader, 
            writer, 
            epoch
        ) 

    if writer is not None:
        writer.close()

    destroy_process_group()