import os
from glob import glob
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
import netCDF4
import numpy as np

class TrainDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        variables: List[str],
        inp_transform_cerra,
        inp_transform_era,
        out_transform,
    ):
        super().__init__()
        self.variables = variables
        self.inp_transform_cerra = inp_transform_cerra
        self.inp_transform_era   = inp_transform_era
        self.out_transform       = out_transform

        era_files   = sorted(glob(os.path.join(root_dir, "era_*.nc")))
        cerra_files = sorted(glob(os.path.join(root_dir, "cerra_*.nc")))

        def make_map(files, prefix: str):
            m = {}
            for f in files:
                key = os.path.basename(f).replace(f"{prefix}_", "").replace(".nc", "")
                m[key] = f
            return m

        era_map   = make_map(era_files,   "era")
        cerra_map = make_map(cerra_files, "cerra")
        common_keys = sorted(set(era_map) & set(cerra_map))

        self.index: List[Tuple[str,str,int]] = []
        for key in common_keys:
            fe = era_map[key]
            fc = cerra_map[key]
            with netCDF4.Dataset(fc) as ds_c:
                T = len(ds_c.dimensions["time"])
            for t in range(T - 1):  # nur bis T-1, damit t+1 existiert
                self.index.append((fe, fc, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fe, fc, t = self.index[idx]

        with netCDF4.Dataset(fe) as ds_e:
            arr_e = np.stack([ds_e[v][t] for v in self.variables], axis=0)  # (V, H_era, W_era)

        with netCDF4.Dataset(fc) as ds_c:
            arr_c_now  = np.stack([ds_c[v][t]   for v in self.variables], axis=0)  # (V, H, W)
            arr_c_next = np.stack([ds_c[v][t+1] for v in self.variables], axis=0)  # (V, H, W)

        inp_c  = self.inp_transform_cerra(torch.from_numpy(arr_c_now ).float())
        inp_e  = self.inp_transform_era(torch.from_numpy(arr_e     ).float())
        target = self.out_transform(torch.from_numpy(arr_c_next).float())

        return inp_c, inp_e, target


class TestDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        variables: List[str],
        inp_transform_cerra,
        inp_transform_era,
        out_transform,
    ):
        super().__init__()
        self.variables = variables
        self.inp_transform_cerra = inp_transform_cerra
        self.inp_transform_era   = inp_transform_era
        self.out_transform       = out_transform

        era_files   = sorted(glob(os.path.join(root_dir, "era_*.nc")))
        cerra_files = sorted(glob(os.path.join(root_dir, "cerra_*.nc")))

        def make_map(files, prefix: str):
            m = {}
            for f in files:
                key = os.path.basename(f).replace(f"{prefix}_", "").replace(".nc", "")
                m[key] = f
            return m

        era_map   = make_map(era_files,   "era")
        cerra_map = make_map(cerra_files, "cerra")
        common_keys = sorted(set(era_map) & set(cerra_map))

        self.index: List[Tuple[str,str,int]] = []
        for key in common_keys:
            fe = era_map[key]
            fc = cerra_map[key]
            with netCDF4.Dataset(fc) as ds_c:
                T = len(ds_c.dimensions["time"])
            for t in range(T - 1):
                self.index.append((fe, fc, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fe, fc, t = self.index[idx]

        with netCDF4.Dataset(fe) as ds_e:
            arr_e = np.stack([ds_e[v][t] for v in self.variables], axis=0)

        with netCDF4.Dataset(fc) as ds_c:
            arr_c_now  = np.stack([ds_c[v][t]   for v in self.variables], axis=0)
            arr_c_next = np.stack([ds_c[v][t+1] for v in self.variables], axis=0)

        inp_c  = self.inp_transform_cerra(torch.from_numpy(arr_c_now ).float())
        inp_e  = self.inp_transform_era(torch.from_numpy(arr_e     ).float())
        target = self.out_transform(torch.from_numpy(arr_c_next).float())

        return inp_c, inp_e, target
