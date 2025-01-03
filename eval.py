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

    if cfg.data.task == 'sub':
        ILP = True
        model_name = 'TripartiteHeteroGNN'
        dataset = ILPDataset(cfg.data.datapath,
                        extra_path=f'{cfg.other.ipm_restarts}restarts_'
                                         f'{cfg.model.params.lappe}lap_'
                                         f'{cfg.other.ipm_steps}steps'
                                         f'{"_upper_" + str(cfg.other.upper) if cfg.other.upper is not None else ""}',
                        upper_bound=cfg.other.upper,
                        rand_starts=cfg.other.ipm_restarts)

    else:
        ILP = False
        model_name = 'TripartiteHeteroGNNClean'
        dataset = LPDataset(cfg.data.datapath,
                        extra_path=f'{cfg.other.ipm_restarts}restarts_'
                                         f'{cfg.model.params.lappe}lap_'
                                         f'{cfg.other.ipm_steps}steps'
                                         f'{"_upper_" + str(cfg.other.upper) if cfg.other.upper is not None else ""}',
                        upper_bound=cfg.other.upper,
                        rand_starts=cfg.other.ipm_restarts,
                        pre_transform=Compose([HeteroAddLaplacianEigenvectorPE(k=cfg.model.params.lappe),
                                                     SubSample(cfg.other.ipm_steps)]))

    data = Datamodule(dataset, cfg.train.batchsize,cfg.data.num_workers,cfg.data.ILP)

    model = Pl_model_wrapper.load_from_checkpoint(cfg.eval.ckpt, 
                                        model_name=model_name,
                                        cfg=cfg,
                                        device=cfg.train.device,
                                        ILP=ILP)
    
    # Initialize trainer and run validation
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1
    )
    
    results = trainer.validate(model, data)
    print(f"Validation results: {results}")
    
if __name__ == '__main__':
    main()