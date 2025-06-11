from lightning import LightningDataModule
import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from typing import Optional, Sequence, Tuple
from dataloader.dataset import TrainDataset, TestDataset


def collate_fn_train(batch):
    inp = torch.stack([item[0] for item in batch])  # [B, V, H, W]
    out = torch.stack([item[1] for item in batch])  # [B, V, H, W]
    return inp, out

def collate_fn_test(batch):
    inp = torch.stack([item[0] for item in batch])
    out = torch.stack([item[1] for item in batch])
    return inp, out

class DLWCDataModule(LightningDataModule):

    def __init__(
        self,
        root_dir,
        variables,
        list_train_intervals,
        batch_size=5,
        test_batch_size=5,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        normalize_mean = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
        normalize_mean = np.concatenate([normalize_mean[v] for v in variables], axis=0)
        normalize_std = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))
        normalize_std = np.concatenate([normalize_std[v] for v in variables], axis=0)

        self.transforms = transforms.Normalize(normalize_mean, normalize_std)

        out_transforms = {}
        normalize_diff_std = dict(np.load(os.path.join(root_dir, "normalize_diff_std_3.npz")))
        normalize_diff_std = np.concatenate([normalize_diff_std[v] for v in variables], axis=0)
        out_transforms[3] = transforms.Normalize(np.zeros_like(normalize_diff_std), normalize_diff_std)
        self.out_transforms = out_transforms

        self.data_train: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def get_lat_lon(self):
        lat = np.load(os.path.join(self.hparams.root_dir, "lat.npy"))
        lon = np.load(os.path.join(self.hparams.root_dir, "lon.npy"))
        return lat, lon
    
    def get_transforms(self):
        return self.transforms, self.out_transforms[3]

    def setup(self, stage: Optional[str] = None):        
        if not self.data_train and not self.data_test:
            self.data_train = TrainDataset(
                root_dir=os.path.join(self.hparams.root_dir, 'train'),
                variables=self.hparams.variables,
                inp_transform=self.transforms,
                out_transform=self.out_transforms[3],
            )

            self.data_test = TestDataset(
                root_dir=os.path.join(self.hparams.root_dir, 'test'),
                variables=self.hparams.variables,
                transform=self.transforms,
            )


    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=collate_fn_train,
            num_workers=4,
        )

    def val_dataloader(self):
        # run validation on the same train‐type collate (or point to your test set)
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.test_batch_size,
            shuffle=False,
            collate_fn=collate_fn_train,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.test_batch_size,
            shuffle=False,
            collate_fn=collate_fn_test,
            num_workers=4,
        )