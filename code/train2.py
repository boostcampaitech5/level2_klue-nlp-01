import sklearn
import torch
import gc
import wandb
import shutil
import numpy as np
import random

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainerCallback,
    TrainingArguments,
    EarlyStoppingCallback,
)

from sklearn.metrics import accuracy_score
from custom.custom_trainer import CustomTrainer
from utils.log import make_log_dirs
from utils.config import *
from load_data import *

NUM_LABELS = 30

class DropoutCallback(TrainerCallback):
    def __init__(self, model, config) -> None:
        super().__init__()
        self.model = model
        self.config = config

    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f'current dropout is {self.model.config.hidden_dropout_prob, self.model.config.attention_probs_dropout_prob}')
        # 원하는 시점에서 Dropout을 변경합니다.
        if state.epoch == self.config['late_dropout_epoch']:
            print(f'dropout change to {self.config["late_hidden_dropout_prob"]}, {self.config["late_attention_probs_dropout_prob"]}')
            self.model.config.hidden_dropout_prob = self.config['late_hidden_dropout_prob']
            self.model.config.attention_probs_dropout_prob = self.config['late_attention_probs_dropout_prob']
            for i in range(self.model.roberta.encoder.layer):
                self.model.roberta.encoder.layer[i].output.dropout.p = self.config['late_hidden_dropout_prob']    
                self.model.roberta.encoder.layer[i].attention.output.dropout.p = self.config['late_attention_probs_dropout_prob']



def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = [
        "no_relation",
        "org:top_members/employees",
        "org:members",
        "org:product",
        "per:title",
        "org:alternate_names",
        "per:employee_of",
        "org:place_of_headquarters",
        "per:product",
        "org:number_of_employees/members",
        "per:children",
        "per:place_of_residence",
        "per:alternate_names",
        "per:other_family",
        "per:colleagues",
        "per:origin",
        "per:siblings",
        "per:spouse",
        "org:founded",
        "org:political/religious_affiliation",
        "org:member_of",
        "per:parents",
        "org:dissolved",
        "per:schools_attended",
        "per:date_of_death",
        "per:date_of_birth",
        "per:place_of_birth",
        "per:place_of_death",
        "org:founded_by",
        "per:religion",
    ]
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(NUM_LABELS)[labels]

    score = np.zeros((NUM_LABELS,))
    for c in range(NUM_LABELS):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(pred):
    """validation을 위한 metrics function을 반환합니다."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)

    return {
        "micro f1 score": f1,
        "auprc": auprc,
        "accuracy": acc,
    }


def train(config=None):
    config=wandb.config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # MODEL_NAME = "klue/roberta-large"
    # MODEL_NAME = "xlm-roberta-large"
    MODEL_NAME = "klue/roberta-small"

    base_config = load_config_temp()
    train_config = base_config.train
    loss_config = base_config.loss

    # config에 my_log 폴더 경로 기록
    folder_name = make_log_dirs(CONFIG.LOGDIR_NAME)
    base_config.folder_dir = folder_name

    # model_name 및 tokenizer 호출
    model_name = base_config.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=None)

    # entity special token를 tokenizer에 추가
    special_token_list = []
    with open("custom/entity_special_token.txt", "r", encoding="UTF-8") as f:
        for token in f:
            special_token_list.append(token.split("\n")[0])
            
    tokenizer.add_special_tokens({"additional_special_tokens": list(set(special_token_list))})

    model_config = AutoConfig.from_pretrained(MODEL_NAME, cache_dir=None)
    model_config.num_labels = 30
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config, cache_dir=None)
    model.resize_token_embeddings(len(tokenizer))
    # model.config.attention_probs_dropout_prob = 0.3
    # model.config.hidden_dropout_prob = 0.3
    model.config.attention_probs_dropout_prob = config['early_attention_probs_dropout_prob']
    model.config.hidden_dropout_prob = config['early_hidden_dropout_prob']
    for i in range(len(model.roberta.encoder.layer)):
        model.roberta.encoder.layer[i].output.dropout.p = config['early_attention_probs_dropout_prob']    
        model.roberta.encoder.layer[i].attention.output.dropout.p = config['early_hidden_dropout_prob']
    # print(config)
    # print(model.config)
    model.to(device)

    # make dataset for pytorch.
    train_dataset, val_dataset, class_num_list = load_train_dataset(
        model_name, base_config["path"], base_config
    )
    
    # breakpoint()
    training_args = TrainingArguments(
        report_to=CONFIG.WANDB,
        seed=base_config.seed,
        output_dir=train_config.output_dir,
        save_total_limit=train_config.save_total_limit,
        save_strategy=train_config.save_strategy,
        per_device_train_batch_size=base_config.batch_size,
        per_device_eval_batch_size=base_config.batch_size,
        logging_dir=train_config.logging_dir,
        logging_steps=train_config.logging_steps,
        evaluation_strategy=train_config.evaluation_strategy,
        load_best_model_at_end=train_config.load_best_model_at_end,
        num_train_epochs=train_config.num_train_epochs,
        learning_rate=train_config.learning_rate,
        warmup_ratio=train_config.warmup_ratio,
        weight_decay=train_config.weight_decay,
        metric_for_best_model=train_config.metric_for_best_model,
        greater_is_better=train_config.greater_is_better,
        fp16=True,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        loss_type=loss_config.loss_type,
        focal_loss_gamma=loss_config.gamma,
        class_num_list=class_num_list,
        max_m=loss_config.max_m,
        weight=loss_config.weight,
        s=loss_config.s,
        device=device,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=train_config.early_stopping_patience)
            # ,DropoutCallback(model, config)
            ]
    )
    
    # train model
    trainer.train()
    # model.save_pretrained(base_config.folder_dir + CONFIG.OUTPUT_PATH)
    # shutil.copyfile(CONFIG.CONFIG_PATH, os.path.join(base_config.folder_dir, CONFIG.CONFIG_NAME))
    return trainer


def objective(config):
    trainer = train(config)
    gc.collect()

    metrics = trainer.evaluate()
    return metrics["eval_micro f1 score"]

def main():
    wandb.init()
    # train(wandb.config)
    score = objective(wandb.config)
    wandb.log({"eval_micro f1 score": score})


sweep_config = {
    # "name": "wandb-sweep",
    "method": "random",
    "metric": {
        "name": "eval/micro f1 score",
        "goal": "maximize",
    },
    "parameters": {
        "early_hidden_dropout_prob": {"distribution": "uniform", "min": 0.0, "max": 0.7},
        "early_attention_probs_dropout_prob": {"distribution": "uniform", "min": 0.0, "max": 0.7},
        "late_dropout_epoch": {"values":[1, 2, 3]},
        "late_hidden_dropout_prob": {"distribution": "uniform", "min": 0.0, "max": 0.7},
        "late_attention_probs_dropout_prob": {"distribution": "uniform", "min": 0.0, "max": 0.7}
    },
}

# torch, np 설정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

sweep_id = wandb.sweep(sweep_config, project="dropout")
wandb.agent(sweep_id, function=main, count=50)
wandb.finish()