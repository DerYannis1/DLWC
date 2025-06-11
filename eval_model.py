import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

from weather_vit_seq2seq import WeatherViTSeq2Seq
from dataloader.dataloader import DLWCDataModule

# -----------------------
# Configuration
# -----------------------
CHECKPOINT_PATH = '/path/to/checkpoints/weather-vit-seq2seq.ckpt'
LOG_DIR = '/path/to/tb_logs/weather_seq2seq'
ROOT_DIR = '/path/to/data'
VARIABLES = ['temperature', 'wind', 'humidity']
BATCH_SIZE = 1
INDEX = 1000000  # desired sample index
IMG_SIZE = (16, 16)
PATCH_SIZE = 4
EMBED_DIM = 768
DEPTH = 12
NUM_HEADS = 12
OUT_CHANNELS = 1

# -----------------------
# Load model
# -----------------------
model = WeatherViTSeq2Seq.load_from_checkpoint(
    CHECKPOINT_PATH,
    variables=VARIABLES,
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    embed_dim=EMBED_DIM,
    depth=DEPTH,
    num_heads=NUM_HEADS,
    out_channels=OUT_CHANNELS,
)
model.eval()

# -----------------------
# Prepare data
# -----------------------
data_module = DLWCDataModule(
    root_dir=ROOT_DIR,
    variables=VARIABLES,
    list_train_intervals=[(0,1)],
    batch_size=BATCH_SIZE,
    test_batch_size=BATCH_SIZE,
)
data_module.setup()

test_loader = data_module.test_dataloader()

# Get specific sample by index
all_samples = []
for batch in test_loader:
    inp, tgt, vars = batch
    all_samples.append((inp, tgt))
    if len(all_samples) * BATCH_SIZE > INDEX:
        break

# Compute batch and within-batch index
i = INDEX // BATCH_SIZE
j = INDEX % BATCH_SIZE
src_img, tgt_img = all_samples[i]
src = src_img[j:j+1]  # [1, V, H, W]
tgt = tgt_img[j:j+1]

# -----------------------
# Predict
# -----------------------
pred = model(src, tgt)

# -----------------------
# Plot input, prediction, target
# -----------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# For simplicity, plot first variable channel
axes[0].imshow(src[0, 0].cpu().numpy(), cmap='viridis')
axes[0].set_title('Input (t0) Variable 0')
axes[0].axis('off')

axes[1].imshow(pred[0, 0].detach().cpu().numpy(), cmap='viridis')
axes[1].set_title('Prediction (t1)')
axes[1].axis('off')

axes[2].imshow(tgt[0, 0].cpu().numpy(), cmap='viridis')
axes[2].set_title('Ground Truth (t1)')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# -----------------------
# Plot training loss curve
# -----------------------
# Load TensorBoard events
ea = event_accumulator.EventAccumulator(
    LOG_DIR,
    size_guidance={
        event_accumulator.SCALARS: 0,
    }
)
ea.Reload()
train_events = ea.Scalars('train/loss')
val_events = ea.Scalars('val/loss')

train_steps = [e.step for e in train_events]
train_values = [e.value for e in train_events]
val_steps = [e.step for e in val_events]
val_values = [e.value for e in val_events]

plt.figure(figsize=(6,4))
plt.plot(train_steps, train_values, label='Train Loss')
plt.plot(val_steps, val_values, label='Val Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.tight_layout()
plt.show()
