import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from Weather_ViT import WeatherViTSeq2Seq
from dataloader.dataloader import DLWCDataModule

# 1. Reproducibility
seed_everything(42)

# 2. Hyperparameters
root_dir = './data'
variables = [
        # list your 35 channel names in order
        "t_100000","t_92500","t_85000","t_70000","t_50000","t_30000","t_20000",
        "z_100000","z_92500","z_85000","z_70000","z_50000","z_30000","z_20000",
        "u_100000","u_92500","u_85000","u_70000","u_50000","u_30000","u_20000",
        "v_100000","v_92500","v_85000","v_70000","v_50000","v_30000","v_20000",
        "r_100000","r_92500","r_85000","r_70000","r_50000","r_30000","r_20000",
    ]
batch_size = 16
test_batch_size = 16
list_train_intervals = [(0, 1)]  # customize as needed
img_size = (16, 16)
patch_size = 4
embed_dim = 512
depth = 8
num_heads = 8
out_channels = len(variables)
lr = 1e-4
max_epochs = 50

# 3. DataModule
data_module = DLWCDataModule(
    root_dir=root_dir,
    variables=variables,
    list_train_intervals=list_train_intervals,
    batch_size=batch_size,
    test_batch_size=test_batch_size
)

# 4. Model
model = WeatherViTSeq2Seq(
    variables=variables,
    img_size=img_size,
    patch_size=patch_size,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    lr=lr
)

# 5. Callbacks and Logger
checkpoint_callback = ModelCheckpoint(
    monitor='val/loss',
    mode='min',
    save_top_k=3,
    filename='weather-vit-seq2seq-{epoch:02d}-{val/loss:.4f}'
)

lr_monitor = LearningRateMonitor(logging_interval='step')
logger = TensorBoardLogger('tb_logs', name='weather_seq2seq')

# 6. Trainer
trainer = Trainer(
    logger=logger,
    callbacks=[checkpoint_callback, lr_monitor],
    max_epochs=max_epochs,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1 if torch.cuda.is_available() else None,
    precision=16 if torch.cuda.is_available() else 32,
    gradient_clip_val=1.0,
    log_every_n_steps=50
)


# 7. Train
trainer.fit(model, datamodule=data_module)

# 8. Test
trainer.test(model, datamodule=data_module)
