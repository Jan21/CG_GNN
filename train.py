import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import Pl_model_wrapper
import numpy as np
import random
import time
import pickle
from data.datamodule import Datamodule # type: ignore
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf

from data.data_preprocess import HeteroAddLaplacianEigenvectorPE, SubSample
from data.dataset import LPDataset, ILPDataset
from torch_geometric.transforms import Compose

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model_name = cfg.model.model_name

    dataset = ILPDataset(cfg.data.datapath,
                        extra_path=f'{cfg.other.ipm_restarts}restarts_'
                                         f'{cfg.model[model_name].lappe}lap_'
                                         f'{cfg.other.ipm_steps}steps'
                                         f'{"_upper_" + str(cfg.other.upper) if cfg.other.upper is not None else ""}',
                        upper_bound=cfg.other.upper,
                        rand_starts=cfg.other.ipm_restarts)

    data = Datamodule(dataset, cfg.train.batchsize,cfg.data.num_workers)

    model = Pl_model_wrapper(model_name,cfg,device=cfg.train.device)
    
    logger = WandbLogger(project="CG_GNN", name=f"{model_name}")
    
    trainer = pl.Trainer(max_epochs=cfg.train.max_epochs, 
                         logger=logger,
                         accelerator="gpu", devices=1,
                         gradient_clip_val=cfg.train.grad_clip)
    
    trainer.fit(model, data)
    trainer.save_checkpoint(cfg.other.ckpt)

          
if __name__ == '__main__':
    main()
     
