import pickle as pickle
import sklearn
import numpy as np

from sklearn.metrics import accuracy_score
from transformers import AutoConfig, TrainingArguments, EarlyStoppingCallback
from load_data import *
from custom.custom_trainer import CustomTrainer
from custom.custom_model import CustomModel
from constants import CONFIG


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = ['no_relation', 'org:top_members/employees', 'org:members',
                  'org:product', 'per:title', 'org:alternate_names',
                  'per:employee_of', 'org:place_of_headquarters', 'per:product',
                  'org:number_of_employees/members', 'per:children',
                  'per:place_of_residence', 'per:alternate_names',
                  'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
                  'per:spouse', 'org:founded', 'org:political/religious_affiliation',
                  'org:member_of', 'per:parents', 'org:dissolved',
                  'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
                  'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
                  'per:religion']
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(
            targets_c, preds_c)
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
        'micro f1 score': f1,
        'auprc': auprc,
        'accuracy': acc,
    }


def train(config, device):
    train_config = config.train

    # model_name 호출
    model_name = config.model_name

    # make dataset for pytorch.
    train_dataset, val_dataset = load_train_dataset(
        model_name, config['path'], config.tokenizer)

    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(model_name)
    model_config.num_labels = CONFIG.NUM_LABELS

    # model = AutoModelForSequenceClassification.from_pretrained(
    #     model_name, config=model_config)

    model = CustomModel(config=model_config)
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
        weight_decay=train_config.weight_decay
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=train_config.early_stopping_patience)]
    )

    # train model
    trainer.train()
    model.save_pretrained(CONFIG.OUTPUT_PATH)

def custom_train(config, device):
    train_config = config.train

    # model_name 및 tokenizer 호출
    model_name = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # entity special token를 tokenizer에 추가
    special_token_list = []
    with open("custom/entity_special_token.txt", "r", encoding="UTF-8") as f:
        for token in f:
            special_token_list.append(token.split("\n")[0])

    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(set(special_token_list))}
    )

    # make dataset for pytorch.
    train_dataset, val_dataset = load_train_dataset(config['path'], tokenizer, config.tokenizer)

    #setting model hyperparameter
    model_config = AutoConfig.from_pretrained(model_name)
    model_config.num_labels = CONFIG.NUM_LABELS


    # model = CustomModel(config=model_config)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=model_config)
    model.resize_token_embeddings(len(tokenizer))
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
        weight_decay=train_config.weight_decay
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=train_config.early_stopping_patience)]
    )

    # train model
    trainer.train()
    model.save_pretrained(CONFIG.OUTPUT_PATH)