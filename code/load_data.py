import pickle as pickle
import pandas as pd
import torch

from transformers import AutoTokenizer
from constants import CONFIG
from sklearn.model_selection import StratifiedShuffleSplit


class RE_Dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_train_dataset(model_name, path, tokenizer_config):
    """csv 파일을 pytorch dataset으로 불러옵니다."""

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # DataFrame로 데이터셋 읽기
    train_dataset, val_dataset = load_split_data(path.train_path)

    # 데이터셋의 label을 불러옴
    train_label = label_to_num(train_dataset["label"].values)
    val_label = label_to_num(val_dataset["label"].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer, tokenizer_config)
    tokenized_val = tokenized_dataset(val_dataset, tokenizer, tokenizer_config)

    # make dataset for pytorch.
    train_dataset = RE_Dataset(tokenized_train, train_label)
    val_dataset = RE_Dataset(tokenized_train, train_label)

    return train_dataset, val_dataset


def load_test_dataset(dataset_dir, tokenizer, tokenizer_config):
    """test dataset을 불러온 후, tokenizing 합니다."""
    test_dataset = load_data(dataset_dir)
    test_label = list(map(int, test_dataset["label"].values))

    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer, tokenizer_config)
    return test_dataset["id"], tokenized_test, test_label


def label_to_num(label):
    """lable을 pickle에 저장된 dict에 따라 int로 변환합니다."""
    num_label = []
    with open(CONFIG.DICT_LABEL_TO_NUM, "rb") as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def preprocessing_dataset(dataset):
    """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    for i, j in zip(dataset["subject_entity"], dataset["object_entity"]):
        i = i[1:-1].split(",")[0].split(":")[1]
        j = j[1:-1].split(",")[0].split(":")[1]

        subject_entity.append(i)
        object_entity.append(j)
    out_dataset = pd.DataFrame(
        {
            "id": dataset["id"],
            "sentence": dataset["sentence"],
            "subject_entity": subject_entity,
            "object_entity": object_entity,
            "label": dataset["label"],
        }
    )
    return out_dataset


def load_data(dataset_dir):
    """csv 파일을 경로에 맡게 불러 옵니다."""
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset)

    return dataset


def load_split_data(dataset_dir):
    """csv 파일을 경로에 맡게 불러 옵니다."""
    train_data, val_data = split_data(dataset_dir)
    train_dataset = preprocessing_dataset(train_data)
    val_dataset = preprocessing_dataset(val_data)

    return train_dataset, val_dataset


def split_data(dataset_dir):
    """csv 파일을 불러와서 train과 dev로 split합니다."""
    pd_dataset = pd.read_csv(dataset_dir)

    # train과 valid 데이터로 split 하기
    dataset_label = label_to_num(pd_dataset["label"].values)
    split_indices = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    train_indices, val_indices = next(split_indices.split(pd_dataset, dataset_label))

    # split된 인덱스에 따라 추출한 후, 설정 인덱스를 제거하고 기본 인덱스(0,1,2, ... , n)으로 변경
    train_data = pd_dataset.loc[train_indices].reset_index(drop=True)
    val_data = pd_dataset.loc[val_indices].reset_index(drop=True)

    return train_data, val_data


def tokenized_dataset(dataset, tokenizer, tokenizer_config):
    """tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
        temp = ""
        temp = e01 + "[SEP]" + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset["sentence"]),
        return_tensors=tokenizer_config.return_tensors,
        padding=tokenizer_config.padding,
        truncation=tokenizer_config.truncation,
        max_length=tokenizer_config.max_length,
        add_special_tokens=tokenizer_config.add_special_tokens,
    )

    return tokenized_sentences


def num_to_label(label):
    """숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다."""
    origin_label = []
    with open(CONFIG.DICT_NUM_TO_LABEL, "rb") as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label
