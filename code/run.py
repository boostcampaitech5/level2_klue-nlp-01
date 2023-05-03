import os
import torch
import random 
import numpy as np
import argparse

from utils.config import load_config
from utils.log import make_log_dirs
from train import train
from inference import inference

from constants import CONFIG

def main(args):
    # config 파일 호출
    config = load_config(args)

    # log 폴더를 생성하는 코드
    if not os.path.isdir(CONFIG.LOGDIR_PATH):
        os.mkdir(CONFIG.LOGDIR_NAME)

    # folder_name = make_log_dirs(CONFIG.LOGDIR_NAME)

    # config에 my_log 폴더 경로 기록
    # config.folder_dir = folder_name

    if config.do_test:
        inference(config)
    else:
        train(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test", type=bool, default=False)
    parser.add_argument("-e", "--experiment", type=bool, default=True)
    args = parser.parse_args()
    main(args)