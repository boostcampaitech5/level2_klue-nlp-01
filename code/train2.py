import os
import pandas as pd
import torch
import gc
import wandb

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    BertTokenizer,
)
from custom_trainer import FocalLoss, CustomTrainer
from load_data import *
from util_class import *
from util_def import *

wandb.login()


def train(config=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    MODEL_NAME = "klue/roberta-large"
    # MODEL_NAME = "xlm-roberta-large"
    # MODEL_NAME = "klue/roberta-small"

    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)

    RE_train_dataset_split, RE_dev_dataset_split, cls_num_list = load_train_val_dataset_split(
        MODEL_NAME, mode=None
    )

    # RE_train_dataset_split, cls_num_list = load_train_val_dataset_split(
    #     MODEL_NAME, mode="last_dance"
    # )

    logging_step = len(RE_train_dataset_split) // config.batch_size
    total_steps = len(RE_train_dataset_split) / config.batch_size * config.num_train_epochs
    warmup_steps = total_steps * config.warmup_ratio

    # logging_step = len(RE_train_dataset_split) // 16
    # total_steps = len(RE_train_dataset_split) / 16 * 20
    # warmup_steps = total_steps * 0.1

    training_args = TrainingArguments(
        report_to=CONFIG.WANDB,
        seed=config.seed,
        data_seed=42,
        output_dir=train_config.output_dir,
        save_total_limit=train_config.save_total_limit,
        save_strategy=train_config.save_strategy,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        logging_dir=train_config.logging_dir,
        logging_steps=train_config.logging_steps,
        evaluation_strategy=train_config.evaluation_strategy,
        load_best_model_at_end=train_config.load_best_model_at_end,
        num_train_epochs=train_config.num_train_epochs,
        learning_rate=train_config.learning_rate,
        warmup_steps=train_config.warmup_steps,
        weight_decay=train_config.weight_decay,
        metric_for_best_model=train_config.metric_for_best_model,
        greater_is_better=train_config.greater_is_better,
        fp16=True,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=RE_train_dataset_split,
        # eval_dataset=RE_train_dataset_split,
        eval_dataset=RE_dev_dataset_split,
        compute_metrics=compute_metrics,
        device=device,
        focal_loss_gamma=2.0,
        loss_type="ldam",
        cls_num_list=cls_num_list,
        # max_m=config.max_m,
        # weight=config.weight,
        # s=config.s,
        max_m=0.5,
        weight=1.0,
        s=35,
        callbacks=[
            CustomEarlyStoppingCallback(
                early_stopping_patience_f1=3, early_stopping_patience_loss=3
            )
        ],
    )

    # train model
    trainer.train()
    model.save_pretrained("./best_model")
    return trainer


# def main():
#     train()


def objective(config):
    trainer = train(config)
    gc.collect()

    metrics = trainer.evaluate()
    return metrics["eval_micro f1 score"]


def main():
    wandb.init()
    score = objective(wandb.config)
    wandb.log({"eval_micro f1 score": score})


sweep_config = {
    "name": "wandb-sweep",
    "method": "bayes",
    "metric": {
        "name": "eval_micro f1 score",
        "goal": "maximize",
    },
    "parameters": {
        # "gamma": {"values": [0, 1, 2, 5]},
        "batch_size": {"values": [32]},
        "learning_rate": {"values": [1e-5, 2e-5, 3e-5]},
        "num_train_epochs": {"values": [10]},
        "weight_decay": {"values": [0.0, 0.01]},
        "warmup_ratio": {"values": [0.2, 0.5]},
        # "max_m": {"min": 0.5, "max": 0.5},
        # "weight": {"min": 0.8, "max": 1.1},
        # "s": {"min": 25, "max": 35},
        "label_smoothing_factor": {"values": [0.0, 0.1]},
    },
}
sweep_id = wandb.sweep(sweep_config, project="dropout")
wandb.agent(sweep_id, function=main, count=10)

if __name__ == "__main__":
    main()
