import pickle as pickle
import pandas as pd
import torch

from collections import Counter
from transformers import AutoTokenizer
from constants import CONFIG
from sklearn.model_selection import StratifiedShuffleSplit
import os
import re


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
    
def compute_class_num_list(labels):
    """class별 비율을 리스트를 반환합니다."""
    counter = Counter(labels)
    class_num_list = [counter[i] for i in range(max(counter.keys())+1)]
    total_count = sum(class_num_list)
    class_num_list = [count / total_count for count in class_num_list]
    return class_num_list

def load_train_dataset(model_name, path, config):
    """csv 파일을 pytorch dataset으로 불러옵니다."""

     # 전처리 전 split된 데이터를 저장하기
    if not os.path.exists(os.path.join(config.folder_dir, path.split_data_dir)):
        os.makedirs(os.path.join(config.folder_dir, path.split_data_dir))
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # DataFrame로 데이터셋 읽기
    train_dataset, val_dataset = load_split_data(path, config.folder_dir)

    # 데이터셋의 label을 불러옴
    train_label = label_to_num(train_dataset["label"].values)
    val_label = label_to_num(val_dataset["label"].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer, config.tokenizer)
    tokenized_val = tokenized_dataset(val_dataset, tokenizer, config.tokenizer)

    # make dataset for pytorch.
    train_dataset = RE_Dataset(tokenized_train, train_label)
    val_dataset = RE_Dataset(tokenized_val, val_label)

    class_num_list = compute_class_num_list(train_label)
    return train_dataset, val_dataset, class_num_list


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
    """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
    type entity_marker
    원본) 이순신은 조선 전기의 무신이다.
    적용 후) @*사람*이순신@은 #^날짜^조선# 전기의 무신이다.
    
    multi-sentence
    원본) 이순신은 조선 전기의 무신이다.
    적용 후) 이순신은 조선 전기의 무신이다. 이 문장에서 조선은 이순신의 날짜이다. 이 때, 이 둘의 관계는
    """

    subject_entity = []
    object_entity = []
    sentences = []
    data_length = len(dataset)

    for idx in range(data_length):
        row = dataset.iloc[idx].to_dict()
        sentence, sbj_data, obj_data = row["sentence"], eval(row["subject_entity"]), eval(row["object_entity"])

        sbj_word, sbj_start_id, sbj_end_id, sbj_type = sbj_data['word'], sbj_data['start_idx'], sbj_data['end_idx'], sbj_data['type']
        obj_word, obj_start_id, obj_end_id, obj_type = obj_data['word'], obj_data['start_idx'], obj_data['end_idx'], obj_data['type']
        
        trans = {"PER": "사람", "ORG": "단체", "DAT": "날짜", "LOC": "위치", "POH": "기타", "NOH": "수량"}

        if sbj_start_id < obj_start_id:
            sentence = sentence[:obj_start_id] + f"@*{trans[obj_type]}*" + obj_word + f"@" + sentence[obj_end_id+1:]
            sentence = sentence[:sbj_start_id] + f"#^{trans[sbj_type]}^" + sbj_word + f"#" + sentence[sbj_end_id+1:]
        else:
            sentence = sentence[:sbj_start_id] + f"#^{trans[sbj_type]}^" + sbj_word + f"#" + sentence[sbj_end_id+1:]
            sentence = sentence[:obj_start_id] + f"@*{trans[obj_type]}*" + obj_word + f"@" + sentence[obj_end_id+1:]

        sentence = sentence + f'이 문장에서 {obj_word}는 {sbj_word}의 {trans[obj_type]}이다. 이 때, 이 둘의 관계는'


        subject_entity.append(sbj_word)
        object_entity.append(obj_word)
        sentences.append(sentence)

    #breakpoint()
    out_dataset = pd.DataFrame({'id': dataset['id'], 'sentence': sentences,
                               'subject_entity': subject_entity, 'object_entity': object_entity, 'label': dataset['label'], })
    
    # 전처리 -> 성능향상이 확실한, UNK로 표시되는 특수문자만 변경
    sentence = out_dataset['sentence'].values

    for i in range(len(sentence)):
        ## 문자 변경
        # sentence[i] = re.sub('[\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002FFFF]+', '[한자]', sentence[i])
        # sentence[i] = re.sub('[\u4e00-\u9fff\u30a0-\u30ff\u3040-\u309f]+', '[일본어]', sentence[i])
        # sentence[i] = re.sub('[\u0370-\u03FF&&[^A-Za-z]]+', '[그리스어]', sentence[i])
        sentence[i] = re.sub('[–]', '-', sentence[i])
        sentence[i] = re.sub('[∼～]', '~', sentence[i])
        # sentence[i] = re.sub('[（「]', '(', sentence[i])
        # sentence[i] = re.sub('[）」]', ')', sentence[i])
        sentence[i] = re.sub('[？]', '?', sentence[i])
        sentence[i] = re.sub('[»]', '<', sentence[i])
        sentence[i] = re.sub('[«]', '>', sentence[i])
        # sentence[i] = re.sub('[»《〈]', '<', sentence[i])
        # sentence[i] = re.sub('[«》〉]', '>', sentence[i])
        # sentence[i] = re.sub('[‘’]', '\'', sentence[i])
        # sentence[i] = re.sub('[“”]', '\"', sentence[i])
        # sentence[i] = re.sub('\(주\)', '㈜', sentence[i])
        sentence[i] = re.sub('€', 'e', sentence[i])
        
        ## 특수문자, 한글, 영어 제외한 문자 제거
        # sentence[i] = re.sub('[^\uAC00-\uD7AF\u1100-\u11FFa-zA-Z0-9{P}\s\(\)\,\~\.\:\"\'\-\·\[\]\/\;\!\?\|\&\*\★\<\>\（\「\）\」\《\〈\‘\’\“\”]+', '[문자]', sentence[i])
        
        ## 문자 제거 후 작업
        # sentence[i] = re.sub('[(][\s\/\~]*[,]*[\s\/\~]*[)]', '', sentence[i]) # (,) 제거
        # sentence[i] = re.sub('[(][\s]*[\,\.\:\!\;\/]+[\s]*', '(', sentence[i])
        # sentence[i] = re.sub('[(][\s]*[\,\.\:\!\;\/]+[\s]*', '(', sentence[i])
        # sentence[i] = re.sub('[(][\s]+', '(', sentence[i]) # 공백제거
        # sentence[i] = re.sub('[\s]+[)]', ')', sentence[i]) # 공백제거
        # sentence[i] = re.sub('[\s]{2,}', ' ', sentence[i]) # 띄어쓰기 2번 이상 제거
        
    out_dataset['sentence'].update(sentence)

    return out_dataset



def load_data(dataset_dir):
    """csv 파일을 경로에 맡게 불러 옵니다."""
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset)

    return dataset


def load_split_data(dataset_dir, save_path):
    """csv 파일을 경로에 맡게 불러 옵니다."""
    train_data, val_data = split_data(dataset_dir, save_path)
    train_dataset = preprocessing_dataset(train_data)
    val_dataset = preprocessing_dataset(val_data)

    train_dataset.to_csv(os.path.join(save_path, dataset_dir.split_preprocess_train_path), index=False)
    val_dataset.to_csv(os.path.join(save_path, dataset_dir.split_preprocess_val_path), index=False)

    return train_dataset, val_dataset


def split_data(dataset_dir, save_path):
    """csv 파일을 불러와서 train과 dev로 split합니다."""
    pd_dataset = pd.read_csv(dataset_dir.train_path)

    # train과 valid 데이터로 split 하기
    dataset_label = label_to_num(pd_dataset["label"].values)
    split_indices = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    train_indices, val_indices = next(split_indices.split(pd_dataset, dataset_label))

    # split된 인덱스에 따라 추출한 후, 설정 인덱스를 제거하고 기본 인덱스(0,1,2, ... , n)으로 변경
    train_data = pd_dataset.loc[train_indices].reset_index(drop=True)
    val_data = pd_dataset.loc[val_indices].reset_index(drop=True)

    train_data.to_csv(os.path.join(save_path, dataset_dir.split_nopreprocess_train_path), index=False)
    val_data.to_csv(os.path.join(save_path, dataset_dir.split_nopreprocess_val_path), index=False)

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
