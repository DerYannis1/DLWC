# visualize_t100000.py

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataloader.dataloader import DLWCDataModule
from ViT import WeatherViT

# ─── Configuration ─────────────────────────────────────────────────────────────
ROOT_DIR     = "data"
CKPT_PATH    = "lightning_logs/version_2/checkpoints/epoch=49-step=15150.ckpt"
VARS = [
        # list your 35 channel names in order
        "t_100000","t_92500","t_85000","t_70000","t_50000","t_30000","t_20000",
        "z_100000","z_92500","z_85000","z_70000","z_50000","z_30000","z_20000",
        "u_100000","u_92500","u_85000","u_70000","u_50000","u_30000","u_20000",
        "v_100000","v_92500","v_85000","v_70000","v_50000","v_30000","v_20000",
        "r_100000","r_92500","r_85000","r_70000","r_50000","r_30000","r_20000",
    ]
TARGET_VAR   = "t_100000"
BATCH_IDX    = 0
FIG1_OUT     = "t100000_vis.png"
FIG2_OUT     = "loss_curve.png"

# ─── Load mean/std for de‐normalization ─────────────────────────────────────────
mean_dict = dict(np.load(os.path.join(ROOT_DIR, "normalize_mean.npz")))
std_dict  = dict(np.load(os.path.join(ROOT_DIR, "normalize_std.npz")))
mean_vec  = np.concatenate([mean_dict[v] for v in VARS], axis=0)
std_vec   = np.concatenate([std_dict[v]  for v in VARS], axis=0)

chan_idx = VARS.index(TARGET_VAR)

# ─── Prepare DataModule & Model ────────────────────────────────────────────────
dm = DLWCDataModule(
    root_dir=ROOT_DIR,
    variables=VARS,
    batch_size=1,
    test_batch_size=1,
    num_workers=0,
)
dm.setup()

model = WeatherViT.load_from_checkpoint(
    CKPT_PATH,
    in_channels=len(VARS),
    out_channels=len(VARS),
)
model.eval()

# ─── Fetch one sample ──────────────────────────────────────────────────────────
import random

# Get the dataset directly and sample a random index
val_dataset = dm.val_dataloader().dataset
rand_idx = random.randint(0, len(val_dataset) - 1)
inp, target, _ = val_dataset[rand_idx]

# Add batch dimension
inp = inp.unsqueeze(0)
target = target.unsqueeze(0)

# ─── Inference ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inp = inp.to(device)

with torch.no_grad():
    pred = model(inp)


# ─── Denormalize & extract channel ────────────────────────────────────────────
def denorm(arr, cidx):
    return (arr[cidx] * std_vec[cidx] + mean_vec[cidx])

inp_f    = denorm(inp[0].cpu().numpy(),    chan_idx)
pred_f   = denorm(pred[0].cpu().numpy(),   chan_idx)
target_f = denorm(target[0].cpu().numpy(), chan_idx)

# ─── Figure 1: field visualization ────────────────────────────────────────────
fig, axs = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
for ax, fld, title in zip(
    axs,
    [inp_f, pred_f, target_f],
    ["Input at t", "Pred at t+1", "Actual at t+1"],
):
    im = ax.imshow(fld, origin="lower")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

fig.suptitle(f"Variable {TARGET_VAR}")
plt.savefig(FIG1_OUT, dpi=200)
print(f"Saved field visualization → {FIG1_OUT}")

# ─── Figure 2: loss curves ─────────────────────────────────────────────────────
# read the CSV from the same logger you used in train.py
metrics_csv = "lightning_logs/version_2/metrics.csv"
df = pd.read_csv(metrics_csv)

# drop any rows where train/loss or val/loss is NaN
df = df.dropna(subset=["train/loss"])

plt.figure(figsize=(6,4), tight_layout=True)
plt.plot(df["epoch"], df["train/loss"], label="train/loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(FIG2_OUT, dpi=200)
print(f"Saved loss curve → {FIG2_OUT}")
