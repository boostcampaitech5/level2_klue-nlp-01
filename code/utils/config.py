from omegaconf import OmegaConf
from constants import CONFIG

def load_config(args):
    """config.yaml과 args를 불러서 config로 반환합니다."""
    config = OmegaConf.load(CONFIG.CONFIG_PATH)
    config.do_inference = args.inference

    return config
