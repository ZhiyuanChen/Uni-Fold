from chanfig import Config

from .dta import DataConfig
from .model import ModelConfig
from .loss import LossConfig
from .constants import chunk_size, d_pair, d_msa, d_template, d_extra_msa, d_single, eps, inf, max_recycling_iters

class BaseConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.globals = GlobalConfig()
        self.data = DataConfig()
        self.model = ModelConfig()
        self.loss = LossConfig()


class GlobalConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.block_size = None
        self.d_pair = d_pair
        self.d_msa = d_msa
        self.d_template = d_template
        self.d_extra_msa = d_extra_msa
        self.d_single = d_single
        self.eps = eps
        self.inf = inf
        self.max_recycling_iters = max_recycling_iters
        self.alphafold_original_mode = False
