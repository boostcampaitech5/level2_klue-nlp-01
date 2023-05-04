# import yaml
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from constants import CONFIG
from argparse import Namespace

# import omegaconf
def load_config(args: Namespace) -> DictConfig:
    """_summary_

    Args:
        args (Namespace): arg

    Returns:
        DictConfig: _description_
    """    
    config = OmegaConf.load(CONFIG.CONFIG_PATH)
    config.do_inference = args.inference
    
    return config
