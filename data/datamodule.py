from collections import defaultdict
import pytorch_lightning as pl
#from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader

from data.utils import collate_fn_ip


class Datamodule(pl.LightningDataModule):
    def __init__(self, dataset, cfg):
        super(Datamodule, self).__init__()
        self.dataset = dataset
        self.batch_size = cfg.data.batch_size
        self.num_workers = cfg.data.num_workers
        self.collate_fn = collate_fn_ip
        self.val_size = cfg.data.num_val_examples
        self.val_batch_size = cfg.data.batch_size
        self.tr_ratio = cfg.data.train_proportion

    def setup(self, stage=None):
        self.train_dataset = self.dataset[:int(len(self.dataset) * self.tr_ratio)]
        self.val_dataset = self.dataset[int(len(self.dataset) * self.tr_ratio):int(len(self.dataset) * self.tr_ratio)+self.val_size]
        self.test_dataset = self.dataset[int(len(self.dataset) * self.tr_ratio)+self.val_size:int(len(self.dataset) * self.tr_ratio)+self.val_size]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=self.num_workers,
                                collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.val_batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.val_batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.collate_fn)
