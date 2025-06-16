import os
from glob import glob
from typing import List
import torch
from torch.utils.data import Dataset
import netCDF4
import numpy as np


class TrainDataset(Dataset):
    def __init__(
        self,
        root_dir,
        variables: List[str],
        inp_transform,
        out_transform,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.variables = variables
        self.inp_transform = inp_transform
        self.out_transform = out_transform

        self.files = sorted(glob(os.path.join(root_dir, "*.nc")))

        # Für jeden Monat: öffne Datei und speichere Anzahl Zeitpunkte
        self.time_index = []
        for f in self.files:
            ds = netCDF4.Dataset(f)
            num_times = len(ds.dimensions["time"])
            self.time_index.extend([(f, i) for i in range(num_times - 1)])  # -1: letzter Punkt hat kein Folgefeld
            ds.close()

    def __len__(self):
        return len(self.time_index)

    def __getitem__(self, index):
        file_path, time_idx = self.time_index[index]
        with netCDF4.Dataset(file_path) as ds:
            data_list_in = [ds[v][time_idx] for v in self.variables]
            data_list_out = [ds[v][time_idx + 1] for v in self.variables]

        inp_data  = np.stack(data_list_in,  axis=0)  # V×H×W
        out_data  = np.stack(data_list_out, axis=0)

        # direkt absolute Werte normalisieren
        inp_tensor = torch.from_numpy(inp_data).float()
        out_tensor = torch.from_numpy(out_data).float()

        return (
            self.inp_transform(inp_tensor),
            self.out_transform(out_tensor),
        )


class TestDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        variables: List[str],
        transform,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.variables = variables
        self.transform = transform
        self.lead_time = 3

        self.files = sorted(glob(os.path.join(root_dir, "*.nc")))

        self.time_index = []
        for f in self.files:
            ds = netCDF4.Dataset(f)
            num_times = len(ds.dimensions["time"])
            self.time_index.extend([(f, i) for i in range(num_times - 1)])  # Nur Paare mit Zielpunkt
            ds.close()

    def __len__(self):
        return len(self.time_index)

    def __getitem__(self, index):
        file_path, time_idx = self.time_index[index]
        with netCDF4.Dataset(file_path) as ds:
            data_list_in = [ds[v][time_idx] for v in self.variables]
            data_list_out = [ds[v][time_idx + 1] for v in self.variables]

        inp_tensor = torch.from_numpy(np.stack(data_list_in, axis=0))
        out_tensor = torch.from_numpy(np.stack(data_list_out, axis=0))

        return (
            self.transform(inp_tensor),
            self.transform(out_tensor),
            self.variables
        )