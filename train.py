import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import Pl_model_wrapper
import numpy as np
import random
import time
import pickle
import os
from data.datamodule import Datamodule # type: ignore
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf

from data.dataset import ProblemsDataset

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    model_name = 'TripartiteHeteroGNN'
    dataset = ProblemsDataset(cfg.data.datapath,
                        extra_path=f'extra_data')


    data = Datamodule(dataset, cfg)

    model = Pl_model_wrapper(model_name,cfg,cfg.train.device)

    if os.path.exists(cfg.train.ckpt) and cfg.train.resume:
        print(f"Loading checkpoint from {cfg.train.ckpt}")
        model = model.load_from_checkpoint(cfg.train.ckpt, 
                                         model_name=model_name,
                                         cfg=cfg,
                                         device=cfg.train.device)
    
    logger = WandbLogger(project="CG_GNN", name=f"{model_name}")
    
    trainer = pl.Trainer(max_epochs=cfg.train.max_epochs, 
                         logger=logger,
                         accelerator="gpu", devices=1,
                         gradient_clip_val=cfg.train.grad_clip)
    
    trainer.fit(model, data)
    trainer.save_checkpoint(cfg.other.ckpt)

          
if __name__ == '__main__':
    main()
     
