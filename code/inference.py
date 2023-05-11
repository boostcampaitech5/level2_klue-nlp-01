from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
from tqdm import tqdm
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


def base_inference(config, device):
    """
      주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    inference_config = config.inference
    
    # 토크나이저 호출
    tokenizer_name = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 저장된 모델 호출
    model_dir = inference_config.model_dir
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.parameters
    model.to(device)

    # test 데이터셋 호출
    test_dataset_dir = config.path.test_path
    test_id, test_dataset, test_label = load_test_dataset(
        test_dataset_dir, tokenizer, config.tokenizer)
    Re_test_dataset = RE_Dataset(test_dataset, test_label)

    # 정답 예측
    pred_answer, output_prob = test(
        model, Re_test_dataset, device, config)  # model에서 class 추론
    pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환.

    # 예측된 정답을 DataFrame으로 저장
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame(
        {'id': test_id, 'pred_label': pred_answer, 'probs': output_prob, })

    # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    output.to_csv(inference_config.output_dir, index=False)
    #### 필수!! ##############################################
    print('---- Finish! ----')
    

def val_inference(config, device):
    """
      주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
      데이터 분석을 위해 val_data를 사용하여 inference를 진행합니다
    """
    inference_config = config.inference
    
    # 토크나이저 호출
    tokenizer_name = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 저장된 모델 호출
    model_dir = inference_config.model_dir
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.parameters
    model.to(device)

    # val 데이터셋 호출
    val_dataset_dir = config.path.val_path
    val_dataset = load_data(val_dataset_dir)
    val_id = val_dataset['id']
    val_label = label_to_num(val_dataset['label'].values)
    tokenized_val = tokenized_dataset(val_dataset, tokenizer, config.tokenizer)
    Re_val_dataset = RE_Dataset(tokenized_val, val_label)

    # 정답 예측
    pred_answer, output_prob = test(
        model, Re_val_dataset, device, config)  # model에서 class 추론
    pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환.

    # 예측된 정답을 DataFrame으로 저장
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame(
        {'id': val_id, 'pred_label': pred_answer, 'probs': output_prob, })
    
    output = pd.merge(val_dataset, output, how='outer', on='id')
    # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    output.to_csv(inference_config.val_output, index=False)
    #### 필수!! ##############################################
    print('---- Finish! ----')
    