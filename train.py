import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from Weather_ViT import WeatherViTSeq2Seq
from dataloader.dataloader import DLWCDataModule

# 1. Reproducibility
seed_everything(42)

# 2. Hyperparameters
root_dir = '/path/to/data'
variables = ['temperature', 'wind', 'humidity']
batch_size = 16
test_batch_size = 16
list_train_intervals = [(0, 1)]  # customize as needed
img_size = (16, 16)
patch_size = 4
embed_dim = 768
depth = 12
num_heads = 12
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
    out_channels=out_channels,
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
    gpus=1 if torch.cuda.is_available() else 0,
    precision=16 if torch.cuda.is_available() else 32,
    gradient_clip_val=1.0,
    log_every_n_steps=50
)

# 7. Train
trainer.fit(model, datamodule=data_module)

# 8. Test
trainer.test(model, datamodule=data_module)
