import pickle as pickle
import sklearn
import numpy as np
import torch
import shutil
import os

from sklearn.metrics import accuracy_score
from transformers import (
    AutoConfig,
    TrainingArguments,
    EarlyStoppingCallback,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from load_data import load_train_dataset
from utils.meta_data import MINOR_LABEL_IDS, REF_SENT, LABEL_TO_ID

from custom.custom_model import CustomModel
from custom.custom_trainer import CustomTrainer
from custom.custom_dataset import my_load_train_dataset, get_ref_inputids
from constants import CONFIG

NUM_LABELS = 30

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


def base_train(config, device):
    """
    custom을 사용하지 않는 베이스 모델의 Train 함수입니다.

    args:
        config : train을 위한 파라미터 정보를 포함합니다.
        device : 학습을 진행할 device 정보를 가지고 있습니다. (CPU/cuda:0)
    return:
        None
    """
    train_config = config.train
    loss_config = config.loss

    # model_name 호출
    model_name = config.model_name

    # make dataset for pytorch.
    train_dataset, val_dataset, class_num_list = load_train_dataset(
        model_name, config["path"], config
    )

    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(model_name)
    model_config.num_labels = CONFIG.NUM_LABELS

    # model = CustomModel(config=model_config)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)
    model.to(device)

    training_args = TrainingArguments(
        report_to=CONFIG.WANDB,
        seed=config.seed,
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
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=train_config.early_stopping_patience)
        ],
    )

    # train model
    trainer.train()
    model.save_pretrained(config.folder_dir + CONFIG.OUTPUT_PATH)
    shutil.copyfile(CONFIG.CONFIG_PATH, os.path.join(config.folder_dir, CONFIG.CONFIG_NAME))


def custom_train(config, device):
    """
    custom_model을 활용해 학습을 진행하기 위한 Train 함수입니다.

    args:
        config : train을 위한 파라미터 정보를 포함합니다.
        device : 학습을 진행할 device 정보를 가지고 있습니다. (CPU/cuda:0)
    return:
        None
    """
    train_config = config.train
    loss_config = config.loss
    NUM_LABELS  = config.num_labels
    
    # model_name 및 tokenizer 호출
    model_name = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ref_labels_id = MINOR_LABEL_IDS[:4]
    # ref_labels_id = sorted(ref_labels_id)
    # ref_sent = [REF_SENT[i] for i in ref_labels_id]
    # ref_input_ids, ref_mask = get_ref_inputids(tokenizer=tokenizer, ref_sent=ref_sent)

    # make dataset for pytorch.
    train_dataset, val_dataset, class_num_list = load_train_dataset(
        model_name, config["path"], config
    )


    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(model_name)
    model_config.num_labels = CONFIG.NUM_LABELS

    # model = RE_Model(config=model_config, n_class=30, ref_input_ids=ref_input_ids, ref_mask=ref_mask, hidden_size=768, PRE_TRAINED_MODEL_NAME=model_name)
    model = CustomModel(model_config=model_config, model_name=model_name, device=device)

    model.to(device)
    # model.init_weights()

    training_args = TrainingArguments(
        report_to=CONFIG.WANDB,
        seed=config.seed,
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
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=train_config.early_stopping_patience)
        ],
    )

    # breakpoint()

    # train model
    trainer.train()
    model.save_pretrained(config.folder_dir + CONFIG.OUTPUT_PATH)
    shutil.copyfile(CONFIG.CONFIG_PATH, os.path.join(config.folder_dir+".", CONFIG.CONFIG_NAME))
