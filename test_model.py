import torch
import matplotlib.pyplot as plt
import numpy as np
from model import SimpleWeatherCNN
from dataloader.dataloader import DLWCDataModule
import pytorch_lightning as pl
import os

def plot_variable(var_tensor, title, vmin=None, vmax=None):
    plt.imshow(var_tensor, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.axis('off')

def main():
    # --- Settings ---
    checkpoint_path = "lightning_logs/version_2/checkpoints/epoch=49-step=60550.ckpt"  # Adjust as needed
    root_dir = "data"
    variables = [
        'z_20000', 'z_30000', 'z_50000', 'z_70000', 'z_85000', 'z_92500', 'z_100000',
        't_20000', 't_30000', 't_50000', 't_70000', 't_85000', 't_92500', 't_100000',
        'u_20000', 'u_30000', 'u_50000', 'u_70000', 'u_85000', 'u_92500', 'u_100000',
        'v_20000', 'v_30000', 'v_50000', 'v_70000', 'v_85000', 'v_92500', 'v_100000',
        'r_20000', 'r_30000', 'r_50000', 'r_70000', 'r_85000', 'r_92500', 'r_100000'
    ]
    var_name = "t_100000"
    time_idx = 0  # Change to desired time index

    # --- Load DataModule and Model ---
    dm = DLWCDataModule(
        root_dir=root_dir,
        variables=variables,
        list_train_intervals=[3],
        batch_size=2,
        test_batch_size=2,
    )
    dm.setup("test")
    test_loader = dm.test_dataloader()

    in_channels = len(variables)
    out_channels = len(variables)
    model = SimpleWeatherCNN.load_from_checkpoint(checkpoint_path, in_channels=in_channels, out_channels=out_channels)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # --- Get a batch ---
    batch = next(iter(test_loader))
    inp, out, batch_vars = batch
    inp = inp.to(device)
    # Find index of t_100000
    var_idx = variables.index(var_name)

    # --- Predict ---
    with torch.no_grad():
        pred = model(inp)

    # --- Plot t_100000 at time t ---
    t0_img = inp[0, var_idx].cpu().numpy()
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plot_variable(t0_img, f"{var_name} at time t")

    # --- Plot predicted t_100000 at time t+1 ---
    pred_img = pred[0, var_idx].cpu().numpy()
    plt.subplot(1, 3, 2)
    plot_variable(pred_img, f"Predicted {var_name} at t+1")

    # --- Plot actual t_100000 at time t+1 ---
    out_img = out[0, var_idx].cpu().numpy()
    plt.subplot(1, 3, 3)
    plot_variable(out_img, f"Actual {var_name} at t+1")

    plt.tight_layout()
    plt.show()
    plt.savefig("t_100000_comparison.png")

    # --- Plot loss curve ---
    # Assumes Lightning logs are in CSV format
    log_path = "lightning_logs/version_2/metrics.csv"
    if os.path.exists(log_path):
        import pandas as pd
        df = pd.read_csv(log_path)
        if "train_loss" in df.columns:
            plt.figure()
            plt.plot(df["step"], df["train_loss"], label="Train Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training Loss Curve")
            plt.legend()
            plt.show()
            plt.savefig("train_loss_curve.png")
        else:
            print("train_loss not found in metrics.csv")
    else:
        print(f"Log file {log_path} not found.")

if __name__ == "__main__":
    main()