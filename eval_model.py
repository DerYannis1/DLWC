import os
import glob
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from model import SimpleWeatherCNN
from dataloader.dataloader import DLWCDataModule
from train import LitWeatherCNN  # <-- Dein LightningModule für das CNN

def plot_loss_curve():
    metrics_path = "./lightning_logs/lightning_logs/version_4/metrics.csv"  # <--- ggf. anpassen!
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

    # 1) Lade die Normalisierungsparameter
    mean_npz = dict(np.load(os.path.join(root_dir, "normalize_mean.npz")))
    std_npz  = dict(np.load(os.path.join(root_dir, "normalize_std.npz")))

    means = np.array([mean_npz[v].item() for v in variables], dtype=float)
    stds  = np.array([std_npz[v].item()  for v in variables], dtype=float)

    # --- Modell laden ---
    ckpt_path = "./lightning_logs/lightning_logs/version_4/checkpoints/last.ckpt"  # <--- anpassen!
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    lit_model = LitWeatherCNN.load_from_checkpoint(ckpt_path, variables=variables, lr=1e-4)
    model = lit_model.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Testdaten holen
    dm = DLWCDataModule(
        root_dir=root_dir,
        variables=variables,
        list_train_intervals=[(0, 1)],
        batch_size=1,
        test_batch_size=1,
    )
    dm.setup()
    batch = next(iter(dm.test_dataloader()))
    # Annahme: batch = (x, y, ...) (wie bei CNN üblich)
    x, y = batch[:2]
    x = x.to(device);  y = y.to(device)

    # Inferenz
    with torch.no_grad():
        y_pred = model(x)

    # Extrahiere das t_100000-Feld
    idx = variables.index("t_100000")
    input_n = x[0,    idx].cpu().numpy()
    pred_n  = y_pred[0, idx].cpu().numpy()
    true_n  = y[0,    idx].cpu().numpy()

    # 2) Inverse Normalize
    input_field = input_n * stds[idx] + means[idx]
    pred_field  = pred_n  * stds[idx] + means[idx]
    true_field  = true_n  * stds[idx] + means[idx]

    # 3) Plotte alle drei in einer Abbildung
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ['Input t_100000 at t', 'Predicted t_100000 at t+1', 'Actual t_100000 at t+1']
    fields = [input_field, pred_field, true_field]

    for ax, field, title in zip(axes, fields, titles):
        im = ax.imshow(field, cmap='coolwarm')
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(im, ax=ax, shrink=0.7)

    plt.tight_layout()
    plt.savefig("Comparison_t_100000_denorm.png")
    plt.close()


if __name__ == "__main__":
    plot_loss_curve()
    plot_weather_sample()
