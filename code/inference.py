from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from load_data import *
import custom.custom_dataset as custom # custom dataset 사용

import os
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
from tqdm import tqdm
# from load_data import num_to_label, load_test_dataset
from custom.custom_dataset import num_to_label, my_load_test_dataset, RE_Dataset

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

     # inference_file을 설정했다면, inference_file을 실행
    if config.inference_file:
        inference_dir = config.inference_file
    else:
        last_log = sorted(os.listdir(CONFIG.LOGDIR_PATH))[-1]
        inference_dir = last_log
    
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
    NUM_LABELS  = config.num_labels
    
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

    # inference_file을 설정했다면, inference_file을 실행
    if config.inference_file:
        inference_dir = config.inference_file
    else:
        last_log = sorted(os.listdir(CONFIG.LOGDIR_PATH))[-1]
        inference_dir = last_log
    
    # 모델 불러오기
    model_dir = f"./logs/{inference_dir}/best_model"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    
    # prediction 폴더 존재 확인
    if not os.path.exists(CONFIG.PREDICTTION_PATH):
        os.makedirs(CONFIG.PREDICTTION_PATH)
    
    ####################################### dev #######################################
    # dev 데이터로 결과 눈으로 확인해보기
    dev_id, dev_sentences, dev_dataset, dev_label = my_load_test_dataset(config.path.val_path, tokenizer, config.tokenizer, NUM_LABELS)
    dev_dataset = RE_Dataset(dev_dataset, dev_label)
    pred_answer, output_prob = test(model, dev_dataset, device, config)  # model에서 class 추론
    output = pd.DataFrame({'id': dev_id, 'sentence' : dev_sentences, 'label' : dev_label, 'pred_label': pred_answer, 'probs': output_prob, })
    
    # prediction 저장 폴더 생성
    if not os.path.exists(os.path.join(inference_config.output_dir, inference_dir)):
        os.makedirs(os.path.join(inference_config.output_dir, inference_dir))
        
    # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    file_path = os.path.join(inference_config.output_dir, inference_dir, f"dev_{inference_config.output_file}")
    output.to_csv(file_path, index=False)
    
    ####################################### Test #######################################
    # test 데이터셋 호출
    test_dataset_dir = config.path.test_path
    test_id, _, test_dataset, test_label = my_load_test_dataset(test_dataset_dir, tokenizer, config.tokenizer, NUM_LABELS)
    test_dataset = RE_Dataset(test_dataset, test_label)

    # 정답 예측
    pred_answer, output_prob = test(model, test_dataset, device, config)  # model에서 class 추론
    # pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환.

    # 예측된 정답을 DataFrame으로 저장
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame({'id': test_id, 'pred_label': pred_answer, 'probs': output_prob, })
    
    # prediction 저장 폴더 생성
    if not os.path.exists(os.path.join(inference_config.output_dir, inference_dir)):
        os.makedirs(os.path.join(inference_config.output_dir, inference_dir))
        
    # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    file_path = os.path.join(inference_config.output_dir, inference_dir, f"test_{inference_config.output_file}")
    output.to_csv(file_path, index=False)
    
    
    #### 필수!! ##############################################
    print('---- Finish! ----')
    

def base_val_inference(config, device):
    """
      주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
      데이터 분석을 위해 val_data를 사용하여 inference를 진행합니다
    """
    inference_config = config.inference
    
    # 토크나이저 호출
    tokenizer_name = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # inference_file을 설정했다면, inference_file을 실행
    if config.inference_file:
        inference_dir = config.inference_file
    else:
        last_log = sorted(os.listdir(CONFIG.LOGDIR_PATH))[-1]
        inference_dir = last_log
        

    # 모델 불러오기
    model_dir = f"./logs/{inference_dir}/best_model"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)

    # val 데이터셋 호출
    val_dataset_dir = os.path.join(f"./logs/{inference_dir}/", config.path.split_nopreprocess_val_path)
    val_dataset = load_data(val_dataset_dir, config.tokenizer)
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
    
    # prediction 폴더 존재 확인
    if not os.path.exists(CONFIG.PREDICTTION_PATH):
        os.makedirs(CONFIG.PREDICTTION_PATH)
        
    # prediction 저장 폴더 생성
    if not os.path.exists(os.path.join(inference_config.output_dir, inference_dir)):
        os.makedirs(os.path.join(inference_config.output_dir, inference_dir))
    
    # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    file_path = os.path.join(inference_config.output_dir, inference_dir, inference_config.val_inference_file)
    output = pd.merge(val_dataset, output, how='outer', on='id')
    output.to_csv(file_path, index=False)
    #### 필수!! ##############################################
    print('---- Finish! ----')
    
    
def custom_val_inference(config, device):
    """
      주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
      데이터 분석을 위해 val_data를 사용하여 inference를 진행합니다
      custom_model을 사용하시는 분들은 알맞게 변경해서 사용해 주세요
    """
    inference_config = config.inference
    
    # 토크나이저 호출
    tokenizer_name = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    ## 변경필요
    # entity special token를 tokenizer에 추가
    special_token_list = []
    with open("custom/entity_special_token.txt", "r", encoding="UTF-8") as f:
        for token in f:
            special_token_list.append(token.split("\n")[0])

    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(set(special_token_list))}
    )
    
    # inference_file을 설정했다면, inference_file을 실행
    if config.inference_file:
        inference_dir = config.inference_file
    else:
        last_log = sorted(os.listdir(CONFIG.LOGDIR_PATH))[-1]
        inference_dir = last_log
        

    # 모델 불러오기
    model_dir = f"./logs/{inference_dir}/best_model"
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)

    ## 변경필요
    # val 데이터셋 호출
    val_dataset_dir = os.path.join(f"./logs/{inference_dir}/", config.path.split_nopreprocess_val_path)
    val_dataset = custom.load_data(val_dataset_dir, config.tokenizer)
    val_id = val_dataset['id']
    val_label = custom.label_to_num(val_dataset['label'].values)
    tokenized_val = custom.tokenized_dataset(val_dataset, tokenizer, config.tokenizer)
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
    
    # prediction 폴더 존재 확인
    if not os.path.exists(CONFIG.PREDICTTION_PATH):
        os.makedirs(CONFIG.PREDICTTION_PATH)
        
    # prediction 저장 폴더 생성
    if not os.path.exists(os.path.join(inference_config.output_dir, inference_dir)):
        os.makedirs(os.path.join(inference_config.output_dir, inference_dir))
    
    # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    file_path = os.path.join(inference_config.output_dir, inference_dir, inference_config.val_inference_file)
    output = pd.merge(val_dataset, output, how='outer', on='id')
    output.to_csv(file_path, index=False)
    #### 필수!! ##############################################
    print('---- Finish! ----')