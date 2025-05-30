import xarray as xr
import numpy as np
import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil


def reshape(input_dir, temp_output_dir):
    input_files = sorted(Path(input_dir).glob("*.nc"))

    variable_map = {
        'var129': 'z',
        'var130': 't',
        'var131': 'u',
        'var132': 'v',
        'var157': 'r',
    }

    target_vars = list(variable_map.values())
    temp_output_dir = Path(temp_output_dir)
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    for file in tqdm(input_files, desc="Reshaping files"):
        ds = xr.open_dataset(file)
        new_vars = {}

        has_var_names = any(var in ds for var in variable_map.keys())

        if has_var_names:
            var_source_map = {variable_map[k]: k for k in variable_map if k in ds}
        else:
            var_source_map = {v: v for v in target_vars if v in ds}

        for var, source_name in var_source_map.items():
            data = ds[source_name]

            if var == "t" and data.max().item() > 200:
                data = data - 273.15

            if "plev" not in data.dims:
                continue

            for i, p in enumerate(ds['plev'].values):
                var_name = f"{var}_{int(p)}"
                new_vars[var_name] = data.isel(plev=i).drop_vars("plev")

        coords = {k: ds.coords[k] for k in ['time', 'lat', 'lon'] if k in ds.coords}
        new_ds = xr.Dataset(new_vars, coords=coords, attrs=ds.attrs)

        output_path = temp_output_dir / file.name.replace(".nc", "_reshaped.nc")
        new_ds.to_netcdf(output_path)
        print(f"Gespeichert: {output_path}")


def normalize(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    files = sorted(input_path.glob("*.nc"))
    if not files:
        print(f"Keine Dateien gefunden in {input_path}")
        return

    datasets = [xr.open_dataset(f) for f in files]
    combined_ds = xr.concat(datasets, dim='time')

    variables = list(combined_ds.data_vars)
    means = {}
    stds = {}

    for var in variables:
        data = combined_ds[var]
        mean = data.mean().item()
        std = data.std().item()
        means[var] = mean
        stds[var] = std
        print(f"{var}: mean={mean}, std={std}")

    normalized_vars = {
        var: (combined_ds[var] - means[var]) / stds[var] for var in variables
    }

    normalized_ds = xr.Dataset(
        data_vars=normalized_vars,
        coords=combined_ds.coords
    )

    normalized_ds.to_netcdf(output_path / "cerra_2019_2021_norm.nc")

    with open(output_path / "mean_dev.json", 'w') as f:
        json.dump({v: {'mean': means[v], 'std': stds[v]} for v in variables}, f, indent=4)

    np.savez(output_path / "normalize_mean.npz", **{k: np.array([v]) for k, v in means.items()})
    np.savez(output_path / "normalize_std.npz", **{k: np.array([v]) for k, v in stds.items()})

    print("Normalisierung abgeschlossen")

    normalize_differences(combined_ds, output_path)


def normalize_differences(dataset, output_path, epsilon=1e-6):
    """Erzeugt normalize_diff_std_3.npz mit Fallback bei std=0."""
    diff_stds = {}
    for var in dataset.data_vars:
        data = dataset[var]
        if "time" not in data.dims:
            continue
        diff = data.isel(time=slice(1, None)) - data.isel(time=slice(0, -1))
        std = diff.std().item()

        if std == 0.0:
            print(f"std von '{var}' war 0 - setze Fallback auf {epsilon}")
        diff_stds[var] = max(std, epsilon)

        print(f"{var} (Diff 3h): std={diff_stds[var]:.6f}")

    np.savez(output_path / "normalize_diff_std_3.npz", **{k: np.array([v]) for k, v in diff_stds.items()})
    print("\n normalize_diff_std_3.npz erfolgreich erstellt.")



def split_and_move_files(reshaped_dir, final_output_dir, train_ratio=0.9):
    reshaped_dir = Path(reshaped_dir)
    final_output_dir = Path(final_output_dir)
    train_dir = final_output_dir / "train"
    test_dir = final_output_dir / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(reshaped_dir.glob("*.nc"))
    num_train = int(len(all_files) * train_ratio)

    train_files = all_files[:num_train]
    test_files = all_files[num_train:]

    for f in train_files:
        shutil.move(str(f), train_dir / f.name)
    for f in test_files:
        shutil.move(str(f), test_dir / f.name)

    print(f"Train: {len(train_files)} Dateien")
    print(f"Test: {len(test_files)} Dateien")


def parse_args():
    parser = argparse.ArgumentParser(description='Reshape and normalize NetCDF files.')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing input data.')
    parser.add_argument('--save_dir', type=str, required=True, help='Final directory for normalized + split data.')
    return parser.parse_args()


def main():
    args = parse_args()

    temp_reshaped_dir = Path(args.save_dir) / "_temp_reshaped"
    reshape(args.root_dir, temp_reshaped_dir)

    normalize(temp_reshaped_dir, Path(args.save_dir))
    split_and_move_files(temp_reshaped_dir, Path(args.save_dir))


if __name__ == "__main__":
    main()
