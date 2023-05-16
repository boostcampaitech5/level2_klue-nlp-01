import os
import argparse
import torch
import wandb

import random
import numpy as np

from utils.config import load_config
from utils.log import make_log_dirs
from train import base_train, custom_train
from inference import base_inference

from constants import CONFIG

def main(args):
    # config 파일 호출
    config = load_config(args)

    # torch, np 설정
    SEED = config.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


    # log 폴더를 생성하는 코드
    if not os.path.isdir(CONFIG.LOGDIR_PATH):
        os.mkdir(CONFIG.LOGDIR_NAME)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # config에 my_log 폴더 경로 기록
    folder_name = make_log_dirs(CONFIG.LOGDIR_NAME)
    config.folder_dir = folder_name

    if config.do_inference:
        base_inference(config, device)
    else:
        if args.custom:
            ## wandb 설정
            wandb.init(project="KLUE-RE", name = folder_name)
            custom_train(config, device)
        else:
            base_train(config, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inference", type=bool, default=False)
    parser.add_argument("-c", "--custom", type=bool, default=False)
    args = parser.parse_args()
    main(args)