import pickle as pickle
import pandas as pd
import torch
from tqdm import tqdm

from constants import CONFIG


class RE_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach()
                for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def my_load_train_dataset(path, tokenizer, config, num_labels):
    """ csv 파일을 pytorch dataset으로 불러옵니다."""

    # DataFrame로 데이터셋 읽기
    train_dataset = load_data(path.train_path, num_labels)
    val_dataset = load_data(path.val_path, num_labels)

    # 데이터셋의 label을 불러옴
    # train_label = label_to_num(train_dataset['label'].values)
    # val_label = label_to_num(val_dataset['label'].values)
    train_label = train_dataset['label'].values
    val_label = val_dataset['label'].values

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer, config.tokenizer)
    tokenized_val = tokenized_dataset(val_dataset, tokenizer, config.tokenizer)

    # make dataset for pytorch.
    train_dataset = RE_Dataset(tokenized_train, train_label)
    val_dataset = RE_Dataset(tokenized_val, val_label)

    return train_dataset, val_dataset


def label_to_num(label):
    """lable을 pickle에 저장된 dict에 따라 int로 변환합니다."""
    num_label = []
    with open(CONFIG.DICT_LABEL_TO_NUM, 'rb') as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def preprocessing_dataset(dataset, num_labels):
    """
        처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
        
        dataset의 entity 데이터를 확인해 Type에 따라 special 토큰을 추가합니다.
        PER(사람), ORG(조직), DAT(시간), LOC(장소), POH(기타 표현), NOH(기타 수량 표현)
        
        ex) 이 돈가스집은 <O:PER>백종원</O:PER> <S:ORG>더본코리아</S:ORG> 대표 ...
    """
    subject_entity = []
    object_entity = []
    sentences = []
    labels = []
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

        sentence = sentence + f'이 문장에서 {sbj_word}는 {obj_word}의 {trans[sbj_type]}이다. 이 때, 이 둘의 관계는'
        
        ## num_labels 값에 따라 라벨 구분
        if num_labels == 2:
            # relation : 0, no_relation : 1
            try:
                if row["label"] == "no_relation":
                    new_label = 1
                else:
                    new_label = 0
            except AttributeError:
                new_label = 0
        elif num_labels == 3:
            # no_relation : 0, per : 1, org : 2
            try:
                if row["label"].startswith("per"):
                    new_label = 1
                elif row["label"].startswith("org"):
                    new_label = 2
                else:
                    new_label = 0
            except AttributeError:
                new_label = 0
        else:
            new_label = row["label"]

        subject_entity.append(sbj_word)
        object_entity.append(obj_word)
        sentences.append(sentence)
        labels.append(new_label)
        
    out_dataset = pd.DataFrame({'id': dataset['id'], 'sentence': sentences,
                               'subject_entity': subject_entity, 'object_entity': object_entity, 'label': labels, })
    return out_dataset


def load_data(dataset_dir, num_labels):
    """ csv 파일을 경로에 맡게 불러 옵니다. """
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset, num_labels)

    return dataset


def tokenized_dataset(dataset, tokenizer, tokenizer_config):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
        temp = ''
        temp = e01 + '[SEP]' + e02
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors=tokenizer_config.return_tensors,
        padding=tokenizer_config.padding,
        truncation=tokenizer_config.truncation,
        max_length=tokenizer_config.max_length,
        add_special_tokens=tokenizer_config.add_special_tokens
    )

    return tokenized_sentences

'''
    #####    Inference     #####
'''
def my_load_test_dataset(path, tokenizer, tokenizer_config, num_labels):
    """test dataset을 불러온 후, tokenizing 합니다."""
    test_dataset = load_data(path, num_labels)
    test_label = list(map(int, test_dataset['label'].values))

    # tokenizing dataset
    tokenized_test = tokenized_dataset(
        test_dataset, tokenizer, tokenizer_config)
    return test_dataset['id'], test_dataset['sentence'], tokenized_test, test_label


def num_to_label(label):
    """숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다."""
    origin_label = []
    with open(CONFIG.DICT_NUM_TO_LABEL, 'rb') as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label
