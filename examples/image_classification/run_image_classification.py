#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import logging
import os
import sys
import json

from typing import List
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset
from torchvision.transforms import Compose, Lambda, ToTensor

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from augment import augmentation_generator, build_eval_transform

import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    SMAForSequenceClassification,
    SMAConfig,
)

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import evaluate

""" Pre-training a 🤗 ViT model as an MAE (masked autoencoder), as proposed in https://arxiv.org/abs/2111.06377."""

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/image-pretraining/requirements.txt")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default="cifar10", metadata={"help": "Name of a dataset from the datasets package"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_split: Optional[str] = field(
        default="train",
    )
    eval_split: Optional[str] = field(
        default="validation",
    )
    image_column_name: Optional[str] = field(
        default="image", metadata={"help": "The column name of the images in the files."}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the training data."})
    validation_dir: Optional[str] = field(default=None, metadata={"help": "A folder containing the validation data."})
    train_val_split: Optional[float] = field(
        default=0.15, metadata={"help": "Percent to split off of train for validation."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    image_width: int = field(
        default=224,
    )
    image_height: int = field(
        default=224,
    )
    image_mean_std: str = field(
        default="imagenet",
    )
    resize_method: str = field(
        default="resize",
    )
    mixup: float = field(
        default=0.8,
    )
    cutmix: float = field(
        default=1.0,
    )
    cutmix_minmax: Optional[List[float]] = field(
        default=None, 
    )
    mixup_prob: float = field(
        default=1.0,
    )
    mixup_switch_prob: float = field(
        default=0.5,
    )
    mixup_mode: float = field(
        default=0.8,
    )
    crop_type: str = field(
        default="rrc"
    )
    color_jitter: float = field(
        default=0.3
    )
    num_classes: int = field(
        default=None,
    )
    three_aug: bool = field(
        default=False,
    )
    no_aug: bool = field(
        default=False,
    )
    eval_crop_ratio: float = field(
        default=0.875,
    )
    random_flip_prob: float = field(
        default=0.5,
    )
    def __post_init__(self):
        data_files = dict()
        if self.train_dir is not None:
            data_files["train"] = self.train_dir
        if self.validation_dir is not None:
            data_files["val"] = self.validation_dir
        self.data_files = data_files if data_files else None


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/feature extractor we are going to pre-train.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name_or_path"}
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    feature_extractor_name: str = field(default=None, metadata={"help": "Name or path of preprocessor config."})
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    hidden_dropout_prob: float = field(
        default=None,
    )
    attention_dropout_prob: float = field(
        default=None
    )
    drop_path_rate: float = field(
        default=None,
    )
    smoothing: float = field(
        default=0.1,
    )
    bce_loss: bool = field(
        default=False,
    )
@dataclass
class CallbackArguments:   
    kill_steps: Optional[int] = field(
        default=None, metadata={"help": "Will kill run at num steps. Used for hyp search."}
    )
    log_images: bool = field(
        default=False,
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    all_args_dict = {}
    for args in (model_args, data_args, training_args):
        for k, v in asdict(args).items():
            all_args_dict[k] = v
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(training_args.output_dir, "all_args.json"), "w") as f:
        json.dump(all_args_dict, f, indent=4)
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_mae", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Initialize our dataset.
    if training_args.do_train:
        train_dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            split=data_args.train_split
        )
    if training_args.do_eval:
        eval_dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            split=data_args.eval_split
        )
    num_classes = data_args.num_classes if data_args.num_classes is not None else (max(max(eval_dataset["label"]), max(train_dataset["label"])) + 1)

    if training_args.do_train:
        column_names = train_dataset.column_names
    else:
        column_names = eval_dataset.column_names

    if data_args.image_column_name is not None:
        image_column_name = data_args.image_column_name
    elif "image" in column_names:
        image_column_name = "image"
    elif "img" in column_names:
        image_column_name = "img"
    else:
        image_column_name = column_names[0]

    # transformations as done in original MAE paper
    # source: https://github.com/facebookresearch/mae/blob/main/main_pretrain.py

    if data_args.image_mean_std != "none":
        if data_args.image_mean_std == "imagenet":
            image_mean = [0.485, 0.456, 0.406]
            image_std = [0.229, 0.224, 0.225]
        elif data_args.image_mean_std == "compute":
            from tqdm import tqdm
            transform = Compose([Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img), ToTensor(),])
            means = []
            stds = []
            for example in tqdm(train_dataset):
                img = transform(example[image_column_name])
                means.append(img.mean((-1, -2)))
                stds.append(img.std((-1, -2)))
            means = torch.stack(means).mean((0))
            stds = torch.stack(stds).mean((0))
            image_mean = [i.item() for i in means]
            image_std = [i.item() for i in stds]
    else:
        image_mean, image_std = 0, 1
    
    if data_args.no_aug:
        train_transforms = augmentation_generator(
            input_size=data_args.image_height,
            mean=image_mean,
            std=image_std,
            crop_type=data_args.crop_type,
            color_jitter=0.0,
            grayscale_prob=0.0,
            solarization_prob=0.0,
            gaussian_blur_prob=0.0,
            random_flip_prob=0.0,
        )
    elif data_args.three_aug:
        train_transforms = augmentation_generator(
            input_size=(data_args.image_height, data_args.image_width),
            crop_type=data_args.crop_type,
            color_jitter=data_args.color_jitter,
            mean=image_mean,
            std=image_std,
            random_flip_prob=data_args.random_flip_prob,
            grayscale_prob=1.0,
            solarization_prob=1.0,
            gaussian_blur_prob=1.0,
        )
    else:
        train_transforms = augmentation_generator(
            input_size=(data_args.image_height, data_args.image_width),
            crop_type=data_args.crop_type,
            color_jitter=data_args.color_jitter,
            mean=image_mean,
            std=image_std,
            grayscale_prob=0.0,
            solarization_prob=0.0,
            gaussian_blur_prob=0.0,
            random_flip_prob=data_args.random_flip_prob,
        )

    eval_transforms = build_eval_transform(
        input_size=data_args.image_height,
        mean=image_mean,
        std=image_std,
        eval_crop_ratio=data_args.eval_crop_ratio,
    )
    mixup_active = data_args.mixup > 0 or data_args.cutmix > 0. or data_args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=data_args.mixup, cutmix_alpha=data_args.cutmix, cutmix_minmax=data_args.cutmix_minmax,
            prob=data_args.mixup_prob, switch_prob=data_args.mixup_switch_prob, mode=data_args.mixup_mode,
            label_smoothing=model_args.smoothing, num_classes=num_classes
        )
    config = SMAConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    )
    
    config.cross_train_noising_args = None
    config.cross_eval_noising_args = None
    
    if model_args.hidden_dropout_prob is not None:
        config.hidden_dropout_prob = model_args.hidden_dropout_prob
    if model_args.attention_dropout_prob is not None:
        config.attention_dropout_prob = model_args.attention_dropout_prob
    if model_args.drop_path_rate is not None:
        config.drop_path_rate = model_args.drop_path_rate
    
    config.num_labels = num_classes
    config.num_outputs = 1

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif model_args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=model_args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    if model_args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()

    model = SMAForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        loss_fn=criterion
    )

    def train_preprocess(examples):
        """Preprocess a batch of images by applying transforms."""
        examples["inputs"] = [train_transforms(image) for image in examples[image_column_name]]
        examples["attention_mask"] = [torch.ones((inp.shape[-2] * inp.shape[-1],)) for inp in examples["inputs"]]
        examples["training"] = [True for _ in range(len(examples["inputs"]))]
        return examples

    def eval_preprocess(examples):
        examples["inputs"] = [eval_transforms(image).view((config.input_channels, -1)).transpose(0, 1) for image in examples[image_column_name]]
        examples["attention_mask"] = [torch.ones((inp.shape[-2])) for inp in examples["inputs"]]
        examples["training"] = [False for _ in range(len(examples["inputs"]))]
        return examples
    if training_args.do_train:
        train_dataset.set_transform(train_preprocess)

    if training_args.do_eval:
        eval_dataset.set_transform(eval_preprocess)

    def collate_fn(examples):
        inputs = torch.stack([example["inputs"] for example in examples])
        attention_mask = torch.stack([example["attention_mask"] for example in examples])
        labels = torch.LongTensor([example["label"] for example in examples])
        if mixup_active and examples[0]["training"]:
            # Training
            if len(inputs) % 2 == 0:
                inputs, labels = mixup_fn(inputs, labels)
            else:
                labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
            inputs = inputs.view((inputs.shape[0], config.input_channels, -1)).transpose(-1, -2)
        else:
            # Evaluation
            labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
        return {"inputs": inputs, "attention_mask": attention_mask, "labels": labels}

    metric = evaluate.load("accuracy")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=np.argmax(p.label_ids, axis=1))
    training_args.remove_unused_columns = False

    # Initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=None,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)



if __name__ == "__main__":
    main()
