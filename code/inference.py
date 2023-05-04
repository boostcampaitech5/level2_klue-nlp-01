from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm
from omegaconf.dictconfig import DictConfig
from load_data import num_to_label, load_test_dataset


def test(model, tokenized_sent, device, config):
    """
      test dataset을 DataLoader로 만들어 준 후,
      batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(
        tokenized_sent, batch_size=config.batch_size, shuffle=False)
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device),
                token_type_ids=data['token_type_ids'].to(device)
            )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()


def inference(config: DictConfig):
    """
      주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    test_config = config.test
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # load tokenizer
    Tokenizer_NAME = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    # load my model
    MODEL_NAME = test_config.model_dir  # model dir.
    model = AutoModelForSequenceClassification.from_pretrained(
        test_config.model_dir)
    model.parameters
    model.to(device)

    # load test datset
    test_dataset_dir = config.path.test_path
    test_id, test_dataset, test_label = load_test_dataset(
        test_dataset_dir, tokenizer)
    Re_test_dataset = RE_Dataset(test_dataset, test_label)

    # predict answer
    pred_answer, output_prob = test(
        model, Re_test_dataset, device, config)  # model에서 class 추론
    pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환.

    # make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame(
        {'id': test_id, 'pred_label': pred_answer, 'probs': output_prob, })

    # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    output.to_csv(test_config.output_dir, index=False)
    #### 필수!! ##############################################
    print('---- Finish! ----')
