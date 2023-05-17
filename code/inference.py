from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from load_data import *

import os
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
from tqdm import tqdm
# from load_data import num_to_label, load_test_dataset
from custom.custom_dataset import num_to_label, load_test_dataset

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
                input_ids=data["input_ids"].to(device),
                attention_mask=data["attention_mask"].to(device),
                subject_mask=data["subject_mask"].to(device),
                object_mask=data["object_mask"].to(device),
            )
        
        logits = outputs
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.extend(result)
        output_prob.extend([f"[{', '.join(map(str,s))}]" for s in prob])
        
    return output_pred, output_prob

def base_inference(config, device):
    """
      주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    inference_config = config.inference
    
    # 토크나이저 호출
    tokenizer_name = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 마지막으로 실행한 모델 파일을 불러올지 여부 확인
    if config.last_file:
        last_log = sorted(os.listdir(CONFIG.LOGDIR_PATH))[-1]
        inference_dir = last_log
    else:
        inference_dir = config.inference_file
    
    # 모델 불러오기
    model_dir = f"./logs/{inference_dir}/best_model"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
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

    # prediction 폴더 존재 확인
    if not os.path.exists(CONFIG.PREDICTTION_PATH):
        os.makedirs(CONFIG.PREDICTTION_PATH)
    
    # prediction 저장 폴더 생성
    if not os.path.exists(os.path.join(inference_config.output_dir, inference_dir)):
        os.makedirs(os.path.join(inference_config.output_dir, inference_dir))
        
    # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    file_path = os.path.join(inference_config.output_dir, inference_dir, inference_config.output_file)
    output.to_csv(file_path, index=False)
    #### 필수!! ##############################################
    print('---- Finish! ----')

def custom_inference(config, device):
    """
      주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    inference_config = config.inference
    
    # 토크나이저 호출
    tokenizer_name = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # entity special token를 tokenizer에 추가
    special_token_list = []
    with open("custom/entity_special_token.txt", "r", encoding="UTF-8") as f:
        for token in f:
            special_token_list.append(token.split("\n")[0])

    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(set(special_token_list))}
    )

    # 마지막으로 실행한 모델 파일을 불러올지 여부 확인
    if config.last_file:
        last_log = sorted(os.listdir(CONFIG.LOGDIR_PATH))[-1]
        inference_dir = last_log
    else:
        inference_dir = config.inference_file
    
    # 모델 불러오기
    model_dir = f"./logs/{inference_dir}/best_model"    
    model = RBERT(model_name = config.model_name, special_tokens_dict=special_token_list, tokenizer = tokenizer)
    model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin")))
    model.plm.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # test 데이터셋 호출
    test_dataset_dir = config.path.test_path
    test_data, test_dataset, test_label = my_load_test_dataset(test_dataset_dir, tokenizer, config.tokenizer)
    Re_test_dataset = RE_Dataset(test_data, test_dataset, test_label, tokenizer)

    # 정답 예측
    pred_answer, output_prob = test(model, Re_test_dataset, device, config)  # model에서 class 추론
    pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환.

    # 예측된 정답을 DataFrame으로 저장
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame({'id': test_id, 'pred_label': pred_answer, 'probs': output_prob, })

    # prediction 폴더 존재 확인
    if not os.path.exists(CONFIG.PREDICTTION_PATH):
        os.makedirs(CONFIG.PREDICTTION_PATH)
    
    # prediction 저장 폴더 생성
    if not os.path.exists(os.path.join(inference_config.output_dir, inference_dir)):
        os.makedirs(os.path.join(inference_config.output_dir, inference_dir))
        
    # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    file_path = os.path.join(inference_config.output_dir, inference_dir, inference_config.output_file)
    output.to_csv(file_path, index=False)
    #### 필수!! ##############################################
    print('---- Finish! ----')
