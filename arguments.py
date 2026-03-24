from __future__ import annotations

from dataclasses import dataclass, field
import os
import sys

from transformers import HfArgumentParser, TrainingArguments


@dataclass
class ModelArguments:
    in_channels: int = field(
        default=3,
        metadata={"help": "Number of input image channels."},
    )


@dataclass
class DataTrainingArguments:
    dataset_name: str = field(
        default="cifar10",
        metadata={"help": "Dataset name. Only `cifar10` is currently supported."},
    )
    cache_dir: str | None = field(
        default=None,
        metadata={"help": "Optional Hugging Face cache directory for datasets."},
    )
    image_size: int = field(
        default=32,
        metadata={"help": "Image size applied by the preprocessing transform."},
    )
    train_split: str = field(
        default="train",
        metadata={"help": "Training split name."},
    )
    eval_split: str = field(
        default="test",
        metadata={"help": "Evaluation split name."},
    )

    def __post_init__(self) -> None:
        if self.dataset_name != "cifar10":
            raise ValueError(
                f"Dataset {self.dataset_name!r} is not supported. Use `cifar10`."
            )


def parse_args() -> tuple[ModelArguments, DataTrainingArguments, TrainingArguments]:
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    return model_args, data_args, training_args
