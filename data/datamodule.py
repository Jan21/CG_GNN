from collections import defaultdict
import pytorch_lightning as pl
#from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader

from data.utils import collate_fn_ip


class Datamodule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size, num_workers):
        super(Datamodule, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = self.dataset[:int(len(self.dataset) * 0.8)]
        self.val_dataset = self.dataset[int(len(self.dataset) * 0.8):int(len(self.dataset) * 0.9)]
        self.test_dataset = self.dataset[int(len(self.dataset) * 0.9):]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=self.num_workers,
                                collate_fn=collate_fn_ip)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=collate_fn_ip)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=collate_fn_ip)
