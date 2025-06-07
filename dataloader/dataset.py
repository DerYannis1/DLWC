# dataset.py

import os
from glob import glob
from typing import List, Tuple
import netCDF4
import numpy as np
import torch
from torch.utils.data import Dataset

class BaseNCDataset(Dataset):
    """
    Shared logic for Train/Test: loads sequences of length 2 in time.
    Returns (inp, out, variables).
    """
    def __init__(
        self,
        root_dir: str,
        variables: List[str],
        inp_transform,
        out_transform,
    ):
        self.files = sorted(glob(os.path.join(root_dir, "*.nc")))
        self.variables = variables
        self.inp_transform = inp_transform
        self.out_transform = out_transform

        # build flat list of (file, time_idx) pairs, excluding last time
        self.time_index = []
        for f in self.files:
            ds = netCDF4.Dataset(f)
            t_dim = len(ds.dimensions["time"])
            ds.close()
            self.time_index += [(f, i) for i in range(t_dim - 1)]

    def __len__(self) -> int:
        return len(self.time_index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        fpath, t = self.time_index[idx]
        with netCDF4.Dataset(fpath) as ds:
            # read all variables at t and t+1
            arr_in = np.stack([ds[v][t]   for v in self.variables], axis=0)
            arr_out = np.stack([ds[v][t+1] for v in self.variables], axis=0)

        inp = torch.from_numpy(arr_in.astype(np.float32))
        out = torch.from_numpy(arr_out.astype(np.float32))

        return (
            self.inp_transform(inp),
            self.out_transform(out),
            self.variables,
        )

class TrainDataset(BaseNCDataset):
    pass

class TestDataset(BaseNCDataset):
    pass
