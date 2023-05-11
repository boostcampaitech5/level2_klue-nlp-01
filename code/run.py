import os
import argparse
import torch
import wandb

from utils.config import load_config
from utils.log import make_log_dirs
from train import base_train, custom_train
from inference import base_inference, custom_inference

from constants import CONFIG

def main(args):
    # config 파일 호출
    config = load_config(args)

    # log 폴더를 생성하는 코드
    if not os.path.isdir(CONFIG.LOGDIR_PATH):
        os.mkdir(CONFIG.LOGDIR_NAME)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if config.only_inference:
        if args.custom:
            print("run custom inference mode!!")
            custom_inference(config, device)
        else:
            print("run normal inference mode!!")
            base_inference(config, device)
    else:
        # config에 my_log 폴더 경로 기록
        folder_name = make_log_dirs(CONFIG.LOGDIR_NAME)
        config.folder_dir = folder_name
        
        wandb.init(project="KLUE-RE", name = folder_name)
        if args.custom:
            ## wandb 설정
            print("run custom mode!!")
            custom_train(config, device)
            custom_inference(config, device)
        else:
            print("run normal mode!!")
            base_train(config, device)
            base_inference(config, device)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inference", type=str, default=False)
    parser.add_argument("-c", "--custom", type=str, default=False)
    parser.add_argument("-l", "--last_file", type=str, default=False)
    args = parser.parse_args()

    args.custom = True if args.custom == "True" else False
    args.last_file = True if args.last_file == "True" else False
    
    main(args)