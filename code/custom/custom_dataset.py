import pickle as pickle
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np

from transformers import AutoTokenizer
from constants import CONFIG


class RE_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""

    def __init__(self, dataset, pair_dataset, labels, tokenizer):
        self.dataset = dataset
        self.pair_dataset = pair_dataset
        self.subject_entity = dataset["subject_entity_token"]
        self.object_entity = dataset["object_entity_token"]
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        subject_entity = self.subject_entity[idx]
        object_entity = self.object_entity[idx]
        
        subject_entity_mask, object_entity_mask = self.add_entity_mask(
            self.pair_dataset, subject_entity, object_entity
        )
        
        item = {key: val[idx].clone().detach()
                for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['subject_mask'] = subject_entity_mask
        item['object_mask'] = object_entity_mask
        return item

    def __len__(self):
        return len(self.labels)

    def add_entity_mask(self, encoded_dict, subject_entity, object_entity):
        """
        based on special token's coordinate, 
        make attention mask for subject and object entities' location 

        Variables:
        - sentence: 그는 [SUB-ORGANIZATION]아메리칸 리그[/SUB-ORGANIZATION]가 출범한 [OBJ-DATE]1901년[/OBJ-DATE] 당시 .426의 타율을 기록하였다.
        - encoded_dict: ['[CLS]', "'", '[SUB-ORGANIZATION]', '아메리칸', '리그', '[/SUB-ORGANIZATION]', "'", '[SEP]', "'", '[OBJ-DATE]', '190', '##1', '##년', '[/OBJ-DATE]', "'", '[SEP]', '그', '##는', '[SUB-ORGANIZATION]', '아메리칸', '리그', '[/SUB-ORGANIZATION]', '가', '출범', '##한', '[OBJ-DATE]', '190', '##1', '##년', '[/OBJ-DATE]', '당시', '.', '42', '##6', '##의', '타율', '##을', '기록', '##하', '##였', '##다', '.', '[SEP]', ]
        - subject_entity: ['[SUB-ORGANIZATION]', '아메리칸', '리그', '[/SUB-ORGANIZATION]']
        - subject_coordinates: index of the first [SUB-{}] added_special_tokens = [2, 18]
        - subject_entity_mask: [0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...]
        - object_entity: ['[OBJ-DATE]', '190', '##1', '##년', '[/OBJ-DATE]']
        - object_coordinates: index of the first [OBJ-{}] added_special_tokens = [9, 25]
        - object_entity_mask: [0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 0 0 0 0 0  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ...]

        Based on special tokens([SUB-ORGANIZATION], [OBJ-DATE]) for each entities, 1 in attention mask annotates the location of the entity.
        For more description, please refer to https://snoop2head.github.io/Relation-Extraction-Code/
        """

        # initialize entity masks
        subject_entity_mask = np.zeros(256, dtype=int)
        object_entity_mask = np.zeros(256, dtype=int)

        # get token_id from encoding subject_entity and object_entity
        subject_entity_token_ids = self.tokenizer.encode(
            subject_entity, add_special_tokens=False
        )
        object_entity_token_ids = self.tokenizer.encode(
            object_entity, add_special_tokens=False
        )

        # get the length of subject_entity and object_entity
        subject_entity_length = len(subject_entity_token_ids)
        object_entity_length = len(object_entity_token_ids)

        # find coordinates of subject_entity_token_ids based on special tokens
        subject_coordinates = np.where(
            encoded_dict["input_ids"] == subject_entity_token_ids[1]
        )

        # change the subject_coordinates into int type
        subject_coordinates = list(map(int, subject_coordinates[0]))

        # notate the location as 1 in subject_entity_mask
        for subject_index in subject_coordinates:
            subject_entity_mask[
                subject_index : subject_index + subject_entity_length
            ] = 1

        # find coordinates of subject_entity_token_ids based on special tokens
        object_coordinates = np.where(
            encoded_dict["input_ids"] == object_entity_token_ids[1]
        )

        # change the object_coordinates into int type
        object_coordinates = list(map(int, object_coordinates[0]))

        # notate the location as 1 in object_entity_mask
        for object_index in object_coordinates:
            object_entity_mask[object_index : object_index + object_entity_length] = 1

        return torch.Tensor(subject_entity_mask), torch.Tensor(object_entity_mask)


def my_load_train_dataset(path, tokenizer, tokenizer_config):
    """ csv 파일을 pytorch dataset으로 불러옵니다."""

    # DataFrame로 데이터셋 읽기
    train_dataset = load_data(path.train_path)
    val_dataset = load_data(path.val_path)

    # 데이터셋의 label을 불러옴
    train_label = label_to_num(train_dataset['label'].values)
    val_label = label_to_num(val_dataset['label'].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer, tokenizer_config)
    tokenized_val = tokenized_dataset(val_dataset, tokenizer, tokenizer_config)

    # make dataset for pytorch.
    train_dataset = RE_Dataset(train_dataset, tokenized_train, train_label, tokenizer)
    val_dataset = RE_Dataset(val_dataset, tokenized_val, val_label, tokenizer)

    return train_dataset, val_dataset


def label_to_num(label):
    """lable을 pickle에 저장된 dict에 따라 int로 변환합니다."""
    num_label = []
    with open(CONFIG.DICT_LABEL_TO_NUM, 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def preprocessing_dataset(dataset):
    """
        처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
        
        dataset의 entity 데이터를 확인해 Type에 따라 special 토큰을 추가합니다.
        # PER(사람), ORG(조직), DAT(시간), LOC(장소), POH(기타 표현), NOH(기타 수량 표현)
        
        ex) 이 돈가스집은 <O:PER>백종원</O:PER> <S:ORG>더본코리아</S:ORG> 대표 ...
    """
    
    subject_entity = []
    object_entity = []
    subject_entity_token = []
    object_entity_token = []
    sentences = []
    data_length = len(dataset)
    
    for idx in tqdm(range(data_length)):
        row = dataset.iloc[idx].to_dict()
        sentence, sbj_data, obj_data = row["sentence"], eval(row["subject_entity"]), eval(row["object_entity"])
        
        sbj_word, sbj_start_id, sbj_end_id, sbj_type = sbj_data['word'], sbj_data['start_idx'], sbj_data['end_idx'], sbj_data['type']
        obj_word, obj_start_id, obj_end_id, obj_type = obj_data['word'], obj_data['start_idx'], obj_data['end_idx'], obj_data['type']
        
        # entity의 위치에 따라 토큰을 추가하는 순서를 다르게 합니다.
        if sbj_start_id < obj_start_id:
            sentence = sentence[:obj_start_id] + f"<O:{obj_type}>" + obj_word + f"</O:{obj_type}>" + sentence[obj_end_id+1:]
            sentence = sentence[:sbj_start_id] + f"<S:{sbj_type}>" + sbj_word + f"</S:{sbj_type}>" + sentence[sbj_end_id+1:]
        else:
            sentence = sentence[:sbj_start_id] + f"<S:{sbj_type}>" + sbj_word + f"</S:{sbj_type}>" + sentence[sbj_end_id+1:]
            sentence = sentence[:obj_start_id] + f"<O:{obj_type}>" + obj_word + f"</O:{obj_type}>" + sentence[obj_end_id+1:]

        subject_entity.append(sbj_word)
        object_entity.append(obj_word)
        subject_entity_token.append(f"<S:{sbj_type}>" + sbj_word + f"</S:{sbj_type}>")
        object_entity_token.append(f"<O:{obj_type}>" + obj_word + f"</O:{obj_type}>")
        sentences.append(sentence)
        
    out_dataset = pd.DataFrame({'id': dataset['id'], 'sentence': sentences,
                               'subject_entity': subject_entity, 'object_entity': object_entity,
                               'subject_entity_token': subject_entity_token, 'object_entity_token': object_entity_token,
                               'label': dataset['label'], })
    
    return out_dataset


def load_data(dataset_dir):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset)

    return dataset


def tokenized_dataset(dataset, tokenizer, tokenizer_config):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
        temp = e01 + '[SEP]' + e02
        concat_entity.append(temp)
    
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors=tokenizer_config.return_tensors,
        padding = "max_length",           # "max_length"
        truncation=tokenizer_config.truncation,     # True
        max_length=tokenizer_config.max_length,     # 256
        add_special_tokens=tokenizer_config.add_special_tokens
    )
    
    return tokenized_sentences

'''
    #####    Inference     #####
'''
def load_test_dataset(dataset_dir, tokenizer, tokenizer_config):
    """test dataset을 불러온 후, tokenizing 합니다."""
    test_dataset = load_data(dataset_dir)
    test_label = list(map(int, test_dataset['label'].values))

    # tokenizing dataset
    tokenized_test = tokenized_dataset(
        test_dataset, tokenizer, tokenizer_config)
    return test_dataset['id'], tokenized_test, test_label


def num_to_label(label):
    """숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다."""
    origin_label = []
    with open(CONFIG.DICT_NUM_TO_LABEL, 'rb') as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label
