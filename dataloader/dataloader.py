# dataloader.py

import os
import numpy as np
from typing import List, Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from dataloader.dataset import TrainDataset, TestDataset
import torch

class DLWCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        variables: List[str],
        batch_size: int = 8,
        test_batch_size: int = 8,
        num_workers: int = 4,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.variables = variables
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        # load per‐channel mean/std
        means = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
        stds  = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
        mean = np.concatenate([means[v] for v in variables], axis=0).tolist()
        std  = np.concatenate([stds[v]  for v in variables], axis=0).tolist()

        # Only Normalize (data is already 16×16)
        self.inp_transform = Normalize(mean, std)
        self.out_transform = Normalize(mean, std)

        self.train_ds: Optional[TrainDataset] = None
        self.val_ds:   Optional[TestDataset]  = None
        self.test_ds:  Optional[TestDataset]  = None

    def setup(self, stage: Optional[str] = None):
        if self.train_ds is None:
            self.train_ds = TrainDataset(
                root_dir=os.path.join(self.root_dir, "train"),
                variables=self.variables,
                inp_transform=self.inp_transform,
                out_transform=self.out_transform,
            )
            self.val_ds = TestDataset(
                root_dir=os.path.join(self.root_dir, "test"),
                variables=self.variables,
                inp_transform=self.inp_transform,
                out_transform=self.out_transform,
            )
            self.test_ds = self.val_ds

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=lambda b: (
                torch.stack([x[0] for x in b]),
                torch.stack([x[1] for x in b]),
                b[0][2],
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda b: (
                torch.stack([x[0] for x in b]),
                torch.stack([x[1] for x in b]),
                b[0][2],
            ),
        )

    def test_dataloader(self):
        return self.val_dataloader()
