import xarray as xr
import numpy as np
import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm


def reshape(input_dir, output_dir):
    input_files = sorted(Path(input_dir).glob("*.nc"))

    variable_map = {
        'var129': 'z',
        'var130': 't',
        'var131': 'u',
        'var132': 'v',
        'var157': 'r',
    }

    target_vars = list(variable_map.values())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

        output_path = output_dir / file.name.replace(".nc", "_reshaped.nc")
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


def normalize_differences(dataset, output_path):
    """Erzeugt normalize_diff_std_3.npz für 3h Differenzen."""
    diff_stds = {}
    for var in dataset.data_vars:
        data = dataset[var]
        if "time" not in data.dims:
            continue
        diff = data.isel(time=slice(1, None)) - data.isel(time=slice(0, -1))
        std = diff.std().item()
        diff_stds[var] = std
        print(f"{var} (Diff 3h): std={std}")

    np.savez(output_path / "normalize_diff_std_3.npz", **{k: np.array([v]) for k, v in diff_stds.items()})
    print("Differenz-Normalisierung abgeschlossen")


def parse_args():
    parser = argparse.ArgumentParser(description='Reshape and normalize NetCDF files.')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory containing input data.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save normalized data.')
    parser.add_argument('--reshape_dir', type=str, required=True, help='Directory to save reshaped files.')
    return parser.parse_args()


def main():
    args = parse_args()

    reshape(args.root_dir, args.reshape_dir)
    normalize(args.reshape_dir, args.save_dir)


if __name__ == "__main__":
    main()
