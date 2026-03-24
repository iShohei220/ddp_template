from __future__ import annotations

import logging

import numpy as np
from datasets import load_dataset
from transformers import Trainer, default_data_collator, set_seed

from arguments import parse_args
from dataset import build_transform, get_num_labels, validate_dataset
from model import build_model

logger = logging.getLogger(__name__)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).astype(np.float32).mean().item()
    return {"accuracy": accuracy}


def main():
    model_args, data_args, training_args = parse_args()

    if not training_args.do_train and not training_args.do_eval:
        raise ValueError("Specify at least one of --do_train or --do_eval.")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=training_args.get_process_log_level(),
    )

    logger.info("Loading arguments and setting random seed")
    set_seed(training_args.seed)

    train_dataset = None
    eval_dataset = None
    with training_args.main_process_first(desc="dataset load"):
        if training_args.do_train:
            train_dataset = load_dataset(
                data_args.dataset_name,
                split=data_args.train_split,
                cache_dir=data_args.cache_dir,
            ).with_transform(build_transform(data_args.image_size))
            validate_dataset(data_args, model_args, train_dataset, data_args.train_split)
        if training_args.do_eval:
            eval_dataset = load_dataset(
                data_args.dataset_name,
                split=data_args.eval_split,
                cache_dir=data_args.cache_dir,
            ).with_transform(build_transform(data_args.image_size))
            validate_dataset(data_args, model_args, eval_dataset, data_args.eval_split)

    reference_dataset = train_dataset if train_dataset is not None else eval_dataset
    if reference_dataset is None:
        raise ValueError("At least one dataset split must be loaded.")
    model = build_model(model_args, num_labels=get_num_labels(reference_dataset))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
    )

    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
