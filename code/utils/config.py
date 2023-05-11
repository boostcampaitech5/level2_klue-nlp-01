from omegaconf import OmegaConf
from constants import CONFIG

def load_config(args):
    """config.yaml과 args를 불러서 config로 반환합니다."""
    config = OmegaConf.load(CONFIG.CONFIG_PATH)
    
    # inference이 있는지 last_file을 하는지 확인
    if args.inference or args.last_file:
        config.only_inference = True
    else:
        config.only_inference = False
    config.inference_dir = args.inference
    config.last_file = args.last_file
    config.inference_file = args.inference

    return config
