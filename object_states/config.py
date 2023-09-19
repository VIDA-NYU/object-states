from pathlib import Path

from yacs.config import CfgNode as CN


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / 'config'
BASE_CONFIG = CONFIG_DIR / 'base'


_C = CN(new_allowed=True)
cfg.merge_from_file(BASE_CONFIG)
# cfg.freeze()


def get_cfg_defaults():
    return _C#.clone()

def get_cfg(cfg_fname):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_fname)
    return cfg
