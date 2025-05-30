import torch
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from model import SimpleWeatherCNN

# === Einstellungen ===
model_ckpt = "lightning_logs/version_0/checkpoints/epoch=9-step=12110.ckpt"  # Passe ggf. an
input_nc = "data/test/cerra_8_2019_reshaped.nc"                # Eingabedatei
time_idx = 0                                                   # Zeitschritt
device = "cpu"                                                 # oder "cuda" falls verfügbar

# === Variablenliste wie im Training ===
variables = [
    'z_20000', 'z_30000', 'z_50000', 'z_70000', 'z_85000', 'z_92500', 'z_100000',
    't_20000', 't_30000', 't_50000', 't_70000', 't_85000', 't_92500', 't_100000',
    'u_20000', 'u_30000', 'u_50000', 'u_70000', 'u_85000', 'u_92500', 'u_100000',
    'v_20000', 'v_30000', 'v_50000', 'v_70000', 'v_85000', 'v_92500', 'v_100000',
    'r_20000', 'r_30000', 'r_50000', 'r_70000', 'r_85000', 'r_92500', 'r_100000'
]
in_channels = len(variables)
out_channels = len(variables)

# === Lade Modell ===
model = SimpleWeatherCNN(in_channels, out_channels)
state_dict = torch.load(model_ckpt, map_location=device)
if "state_dict" in state_dict:
    model.load_state_dict(state_dict["state_dict"])
else:
    model.load_state_dict(state_dict)
model.eval()
model.to(device)

# === Lade Eingabedaten ===
ds = xr.open_dataset(input_nc)
if "time" in ds.dims:
    arr = np.stack([ds[v].isel(time=time_idx).values for v in variables])
else:
    arr = np.stack([ds[v].values for v in variables])
inp = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)  # [1, C, H, W]

# === Vorhersage ===
with torch.no_grad():
    pred = model(inp)  # [1, C, H, W]
pred = pred.cpu().numpy()[0]  # [C, H, W]

# === Temperatur t_100000 extrahieren ===
idx = variables.index("t_100000")
temp_input = arr[idx]  # [H, W] Eingabe (aktueller Zeitschritt)
temp_pred = pred[idx]  # [H, W] Modellvorhersage

# === Tatsächliches Ergebnis laden (angenommen, Ziel ist t+1) ===
# Hier als Beispiel: t+1 Zeitschritt, passe ggf. an!
if "time" in ds.dims and ds.dims["time"] > time_idx + 1:
    temp_actual = ds["t_100000"].isel(time=time_idx+1).values
else:
    temp_actual = np.full_like(temp_pred, np.nan)  # Fallback, falls nicht vorhanden

# === Plot ===
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
im0 = axs[0].imshow(temp_input, cmap="coolwarm")
axs[0].set_title(f"Eingabe t={time_idx}\nt_100000")
plt.colorbar(im0, ax=axs[0], fraction=0.046)

im1 = axs[1].imshow(temp_pred, cmap="coolwarm")
axs[1].set_title("Vorhersage t+1\nt_100000")
plt.colorbar(im1, ax=axs[1], fraction=0.046)

im2 = axs[2].imshow(temp_actual, cmap="coolwarm")
axs[2].set_title(f"Ziel t={time_idx+1}\nt_100000")
plt.colorbar(im2, ax=axs[2], fraction=0.046)

for ax in axs:
    ax.axis("off")

plt.tight_layout()
plt.savefig("temp_prediction_comparison.png")
plt.show()