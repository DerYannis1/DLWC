import os
import glob
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from DLWC_transformer import DLWCTransformer
from dataloader.dataloader import DLWCDataModule
from train import WeatherForecast


def plot_loss_curve():
    metrics_path = "./lightning_logs/lightning_logs/version_1/metrics.csv"
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}")
    metrics = pd.read_csv(metrics_path)

    train = metrics[metrics['train_loss_epoch'].notnull()]
    val   = metrics[metrics['val_loss'].notnull()]

    plt.figure()
    plt.plot(train['epoch'], train['train_loss_epoch'], label='Train Loss')
    plt.plot(val['epoch'],   val['val_loss'],         label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Train & Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Loss_Curve.png")
    plt.close()


def plot_weather_sample():
    root_dir = './data'
    variables = [
        "t_100000","t_92500","t_85000","t_70000","t_50000","t_30000","t_20000",
        "z_100000","z_92500","z_85000","z_70000","z_50000","z_30000","z_20000",
        "u_100000","u_92500","u_85000","u_70000","u_50000","u_30000","u_20000",
        "v_100000","v_92500","v_85000","v_70000","v_50000","v_30000","v_20000",
        "r_100000","r_92500","r_85000","r_70000","r_50000","r_30000","r_20000",
    ]

    # load separate normalization stats for CERRA (output/input) and ERA
    mean_c = dict(np.load(os.path.join(root_dir, "normalize_mean_cerra.npz")))
    std_c  = dict(np.load(os.path.join(root_dir, "normalize_std_cerra.npz")))
    mean_e = dict(np.load(os.path.join(root_dir, "normalize_mean_era.npz")))
    std_e  = dict(np.load(os.path.join(root_dir, "normalize_std_era.npz")))

    # checkpoint
    ckpt_path = "./lightning_logs/lightning_logs/version_1/checkpoints/last.ckpt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    lit_model = LitWeatherForecast.load_from_checkpoint(ckpt_path)
    model = lit_model.model.to('cuda' if torch.cuda.is_available() else 'cpu').eval()

    # DataModule
    dm = DLWCDataModule(
        root_dir=root_dir,
        variables=variables,
        batch_size=1,
        test_batch_size=1
    )
    dm.setup()
    test_loader = dm.test_dataloader()

    # get one batch: (cerra_now, era_now, cerra_next)
    cerra, era, true = next(iter(test_loader))
    device = next(model.parameters()).device
    cerra, era, true = cerra.to(device), era.to(device), true.to(device)

    # inference
    with torch.no_grad():
        t_int  = torch.ones(cerra.size(0), device=device)  # or your actual lead time
        pred = model(cerra, era, t_int)

    # pick variable slice for plotting
    idx = variables.index("t_100000")

    # build arrays of means/stds
    mean_arr_c = np.array([mean_c[v].item() for v in variables])
    std_arr_c  = np.array([std_c[v].item()  for v in variables])
    mean_arr_e = np.array([mean_e[v].item() for v in variables])
    std_arr_e  = np.array([std_e[v].item()  for v in variables])

    # denormalize fields
    era_field   = (era [0, idx].cpu().numpy() * std_arr_e[idx]) + mean_arr_e[idx]
    cerra_field= (cerra[0, idx].cpu().numpy() * std_arr_c[idx]) + mean_arr_c[idx]
    pred_field = (pred [0, idx].cpu().numpy() * std_arr_c[idx]) + mean_arr_c[idx]
    true_field = (true[0, idx].cpu().numpy() * std_arr_c[idx]) + mean_arr_c[idx]

    # plot 4-panel
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, field, title in zip(
        axes,
        [era_field, cerra_field, pred_field, true_field],
        [
            'ERA input t_100000 at t',
            'CERRA input t_100000 at t',
            'Predicted t_100000 at t+1',
            'Actual CERRA t_100000 at t+1'
        ]
    ):
        im = ax.imshow(field, cmap='coolwarm')
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(im, ax=ax, shrink=0.7)

    plt.tight_layout()
    plt.savefig("Comparison_t_100000.png")
    plt.close()


if __name__ == "__main__":
    plot_loss_curve()
    plot_weather_sample()
