from pytorch_lightning import LightningDataModule
import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from typing import Optional
from dataloader.dataset import TrainDataset, TestDataset


def collate_fn_train_dual(batch):
    # batch: List of (cerra_now, era_now, cerra_next)
    cerra, era, target = zip(*batch)
    cerra = torch.stack(cerra)   # [B, V, H, W]
    era   = torch.stack(era)     # [B, V, H_era, W_era]
    target= torch.stack(target)  # [B, V, H, W]
    return cerra, era, target


def collate_fn_test_dual(batch):
    return collate_fn_train_dual(batch)


class DLWCDataModule(LightningDataModule):

    def __init__(
        self,
        root_dir: str,
        variables: list,
        batch_size: int = 5,
        test_batch_size: int = 5,
    ):
        super().__init__()
        self.root_dir        = root_dir
        self.variables       = variables
        self.batch_size      = batch_size
        self.test_batch_size = test_batch_size

        # --- load separate normalization stats ---
        cerra_mean = dict(np.load(os.path.join(root_dir, "normalize_mean_cerra.npz")))
        cerra_std  = dict(np.load(os.path.join(root_dir, "normalize_std_cerra.npz")))
        era_mean   = dict(np.load(os.path.join(root_dir, "normalize_mean_era.npz")))
        era_std    = dict(np.load(os.path.join(root_dir, "normalize_std_era.npz")))

        # stack in variable order
        m_c = np.concatenate([cerra_mean[v] for v in variables], axis=0)
        s_c = np.concatenate([cerra_std [v] for v in variables], axis=0)
        m_e = np.concatenate([era_mean  [v] for v in variables], axis=0)
        s_e = np.concatenate([era_std  [v] for v in variables], axis=0)

        # transforms: normalize CERRA and ERA separately
        self.cerra_transform = transforms.Normalize(mean=m_c, std=s_c)
        self.era_transform   = transforms.Normalize(mean=m_e, std=s_e)
        # output uses CERRA stats
        self.out_transform   = self.cerra_transform

        self.data_train: Optional[Dataset] = None
        self.data_test:  Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if self.data_train is None and self.data_test is None:
            self.data_train = TrainDataset(
                root_dir=os.path.join(self.root_dir, 'train'),
                variables=self.variables,
                inp_transform_cerra=self.cerra_transform,
                inp_transform_era=self.era_transform,
                out_transform=self.out_transform,
            )
            self.data_test = TestDataset(
                root_dir=os.path.join(self.root_dir, 'test'),
                variables=self.variables,
                inp_transform_cerra=self.cerra_transform,
                inp_transform_era=self.era_transform,
                out_transform=self.out_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn_train_dual,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=collate_fn_test_dual,
            num_workers=4,
        )

    def test_dataloader(self):
        return self.val_dataloader()
