from collections import defaultdict
import pytorch_lightning as pl
#from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader

from data.utils import collate_fn_ip, collate_fn_ilp


class Datamodule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size, num_workers,ILP):
        super(Datamodule, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ILP = ILP
        if ILP:
            self.collate_fn = collate_fn_ilp
            self.val_size = 500
            self.val_batch_size = 64
        else:
            self.collate_fn = collate_fn_ip
            self.val_size = 500
            self.val_batch_size = batch_size

    def setup(self, stage=None):


        self.train_dataset = self.dataset[:int(len(self.dataset) * 0.9)]
        self.val_dataset = self.dataset[int(len(self.dataset) * 0.9):int(len(self.dataset) * 0.9)+self.val_size]
        self.test_dataset = self.dataset[int(len(self.dataset) * 0.9)+self.val_size:int(len(self.dataset) * 0.9)+self.val_size]

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
