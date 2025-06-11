import os
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer, LightningModule, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from torch.optim import Adam

from Weather_ViT import SimpleWeatherTransformer
from dataloader.dataloader import DLWCDataModule

class LitWeatherForecast(LightningModule):
    def __init__(
        self,
        variables,
        img_size,
        patch_size,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio,
        lr,
        list_train_intervals=[(0, 1)],
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = SimpleWeatherTransformer(
            variables=variables,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        # simple MSE loss
        self.loss_fn = F.mse_loss

    def forward(self, x, variables, time_interval):
        return self.model(x, variables, time_interval)

    def _shared_step(self, batch, stage):
        x, y = batch  # x, y: (B, V, H, W)
        B = x.size(0)
        t_int = torch.ones(B, device=x.device) * self.hparams.list_train_intervals[0][1]
        y_hat = self(x, self.hparams.variables, t_int)
        loss = self.loss_fn(y_hat, y)
        self.log(f"{stage}_loss", loss, on_step=(stage=='train'), on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, 'test')

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.hparams.lr)
        return opt


if __name__ == "__main__":
    # 1. Reproducibility
    seed_everything(42)

    # 2. Hyperparameters
    root_dir = './data'
    variables = [
        "t_100000","t_92500","t_85000","t_70000","t_50000","t_30000","t_20000",
        "z_100000","z_92500","z_85000","z_70000","z_50000","z_30000","z_20000",
        "u_100000","u_92500","u_85000","u_70000","u_50000","u_30000","u_20000",
        "v_100000","v_92500","v_85000","v_70000","v_50000","v_30000","v_20000",
        "r_100000","r_92500","r_85000","r_70000","r_50000","r_30000","r_20000",
    ]
    batch_size = 12
    test_batch_size = 12
    list_train_intervals = [(0, 1)]
    img_size = (16, 16)
    patch_size = 4
    embed_dim = 512
    depth = 6
    num_heads = 8
    mlp_ratio = 4.0
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

    # 4. LightningModule
    lit_model = LitWeatherForecast(
        variables=variables,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        lr=lr,
    )

    # 5. Logger & Callbacks
    logger = CSVLogger("lightning_logs")
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', mode='min', save_top_k=3,
        filename='{epoch:02d}'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # 6. Trainer
    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
    )

    # 7. Train & Test
    trainer.fit(lit_model, datamodule=data_module)
    trainer.test(lit_model, datamodule=data_module)
