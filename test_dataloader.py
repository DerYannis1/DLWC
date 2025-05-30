import os
import torch
from dataloader.dataloader import DLWCDataModule
import xarray as xr
from pathlib import Path


def load_all_variables(root_dir):
    train_dir = Path(root_dir) / "train"
    nc_files = sorted(train_dir.glob("*.nc"))

    if not nc_files:
        raise FileNotFoundError(f"❌ Keine NetCDF-Dateien in {train_dir}")

    sample_file = nc_files[0]
    print(f"🔍 Lade Beispiel-Datei: {sample_file.name}")

    ds = xr.open_dataset(sample_file)
    variables = list(ds.data_vars)
    ds.close()

    if not variables:
        raise ValueError("❌ Keine Variablen in der Datei gefunden")

    print(f"📄 Gefundene Variablen: {variables}")
    return variables


def try_train_dataloader(dm):
    print("\n📦 Teste Trainings-Dataloader ...")
    loader = dm.train_dataloader()
    batch = next(iter(loader))

    inp, out, mean, std, interval, var_names = batch

    print("✅ Train Batch geladen")
    print(f"Input shape  : {inp.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Mean shape   : {mean.shape}")
    print(f"Std shape    : {std.shape}")
    print(f"Interval     : {interval.shape}")
    print(f"Variablen    : {var_names}")

    print(f"Input[0,0,0,0] = {inp[0,0,0,0].item():.4f}")
    print(f"Output[0,0,0,0] = {out[0,0,0,0].item():.4f}")


def try_test_dataloader(dm):
    print("\n📦 Teste Test-Dataloader ...")
    loader = dm.test_dataloader()
    batch = next(iter(loader))

    inp, out, var_names = batch

    print("✅ Test Batch geladen")
    print(f"Input shape  : {inp.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Variablen    : {var_names}")

    print(f"Input[0,0,0,0] = {inp[0,0,0,0].item():.4f}")
    print(f"Output[0,0,0,0] = {out[0,0,0,0].item():.4f}")


def main():
    root_dir = "data"
    variables = load_all_variables(root_dir)

    print("\n🔧 Initialisiere DataModule ...")
    dm = DLWCDataModule(
        root_dir=root_dir,
        variables=variables,
        list_train_intervals=[3],
        batch_size=2,
        test_batch_size=2,
    )
    dm.setup()

    try_train_dataloader(dm)
    try_test_dataloader(dm)

    print("\n✅ Train- und Test-Dataloader funktionieren einwandfrei.")


if __name__ == "__main__":
    main()
