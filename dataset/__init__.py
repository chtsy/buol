import sys
from .front3d import Front3D
from .matterport import Matterport

cur_mudule = sys.modules[__name__]
def dataset(cfg):
    return getattr(cur_mudule, cfg.DATASET)(cfg)

