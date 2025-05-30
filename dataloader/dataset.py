import os
from glob import glob
from typing import List
import torch
from torch.utils.data import Dataset
import netCDF4
import numpy as np

def get_data_given_path(path, variables):
    """
    Lädt die angegebenen Variablen aus einer NetCDF-Datei und gibt sie als gestapeltes numpy-Array zurück.
    """
    with netCDF4.Dataset(path, 'r') as f:
        data_list = []
        for v in variables:
            if v not in f.variables:
                raise KeyError(f"Variable '{v}' nicht in Datei {path} gefunden.")
            data_list.append(np.array(f.variables[v][:]))
        return np.stack(data_list, axis=0)


def get_out_path(root_dir, year, inp_file_idx):
    """
    Gibt den Pfad zur nächsten Datei (1 Schritt = 3h) zurück,
    auch über den Jahreswechsel hinweg.
    """
    out_file_idx = inp_file_idx + 1
    out_path = os.path.join(root_dir, f'{year}_{out_file_idx:04}.nc')
    
    if os.path.exists(out_path):
        return out_path

    # Falls Datei im gleichen Jahr nicht existiert -> nächstes Jahr
    next_year = year + 1
    out_path_next_year = os.path.join(root_dir, f'{next_year}_0000.nc')
    
    if os.path.exists(out_path_next_year):
        return out_path_next_year

    raise FileNotFoundError(f"Ausgabedatei nicht gefunden: {out_path} oder {out_path_next_year}")


class TrainDataset(Dataset):
    def __init__(
        self,
        root_dir,
        variables,
        inp_transform,
        out_transform,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.variables = variables
        self.inp_transform = inp_transform
        self.out_transform = out_transform

        file_paths = sorted(glob(os.path.join(root_dir, "*.nc")))
        self.inp_file_paths = file_paths[:-1]  # Nur ein Schritt in die Zukunft

    def __len__(self):
        return len(self.inp_file_paths)

    def __getitem__(self, index):
        inp_path = self.inp_file_paths[index]
        inp_data = get_data_given_path(inp_path, self.variables)

        year, inp_file_idx = os.path.basename(inp_path).split(".")[0].split("_")
        year, inp_file_idx = int(year), int(inp_file_idx)

        out_path = get_out_path(self.root_dir, year, inp_file_idx)
        out_data = get_data_given_path(out_path, self.variables)

        diff = torch.from_numpy(out_data - inp_data)
        inp_tensor = torch.from_numpy(inp_data)

        return (
            self.inp_transform(inp_tensor),     # V x H x W
            self.out_transform(diff),           # V x H x W (nur 3h-Differenz)
            torch.from_numpy(self.out_transform.mean),
            torch.from_numpy(self.out_transform.std),
            torch.tensor([0.3], dtype=torch.float32),  # 3h / 10.0
            self.variables,
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

        self.file_paths = sorted(glob(os.path.join(root_dir, "*.nc")))
        self.inp_file_paths = self.file_paths[:-1]  # letztes Feld hat kein 3h-Ziel mehr

    def __len__(self):
        return len(self.inp_file_paths)

    def __getitem__(self, index):
        inp_path = self.inp_file_paths[index]
        inp_data = get_data_given_path(inp_path, self.variables)

        year, inp_file_idx = os.path.basename(inp_path).split(".")[0].split("_")
        year, inp_file_idx = int(year), int(inp_file_idx)

        # Nur 3 Stunden in die Zukunft
        out_path = get_out_path(self.root_dir, year, inp_file_idx)
        out_data = get_data_given_path(out_path, self.variables)

        inp_tensor = torch.from_numpy(inp_data)
        out_tensor = torch.from_numpy(out_data)

        return (
            self.transform(inp_tensor),    # Input: V x H x W
            self.transform(out_tensor),   # Target: V x H x W (nach 3h)
            self.variables,               # List of variable names
        )
