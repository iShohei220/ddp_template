import torch
from torchvision.transforms import v2

from arguments import DataTrainingArguments, ModelArguments

IMAGE_COLUMN = "img"
LABEL_COLUMN = "label"


def build_transform(image_size: int):
    transform = v2.Compose(
        [
            v2.Resize((image_size, image_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float, scale=True),
        ]
    )

    def apply_transform(examples):
        pixel_values = [transform(image.convert("RGB")) for image in examples[IMAGE_COLUMN]]
        return {"pixel_values": pixel_values, "labels": examples[LABEL_COLUMN]}

    return apply_transform


def get_num_labels(dataset) -> int:
    return dataset.features[LABEL_COLUMN].num_classes


def validate_dataset(
    data_args: DataTrainingArguments,
    model_args: ModelArguments,
    dataset,
    split: str,
) -> None:
    num_labels = get_num_labels(dataset)
    if num_labels is None:
        raise ValueError(f"{split} dataset does not expose class labels")

    example = dataset[0]
    image = example["pixel_values"]
    if image.shape[0] != model_args.in_channels:
        raise ValueError(
            f"{split} dataset channel count ({image.shape[0]}) does not match model ({model_args.in_channels})"
        )
    if image.shape[-2:] != (data_args.image_size, data_args.image_size):
        raise ValueError(
            f"{split} dataset resolution ({image.shape[-2:]}) does not match args ({data_args.image_size})"
        )
