import os
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer, LightningModule, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from torch.optim import Adam

from DLWC_transformer import DLWCTransformer
from dataloader.dataloader import DLWCDataModule

class WeatherForecast(LightningModule):
    def __init__(
        self,
        variables,
        img_size_cerra,
        img_size_era,
        patch_size_cerra,
        patch_size_era,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio,
        lr,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = DLWCTransformer(
            variables        = variables,
            img_size_cerra   = img_size_cerra,
            img_size_era     = img_size_era,
            patch_size_cerra = patch_size_cerra,
            patch_size_era   = patch_size_era,
            embed_dim        = embed_dim,
            depth            = depth,
            num_heads        = num_heads,
            mlp_ratio        = mlp_ratio,
        )
        self.loss_fn = F.mse_loss

    def forward(self, x_cerra, x_era, time_interval):
        return self.model(x_cerra, x_era, time_interval)

    def _shared_step(self, batch, stage):
        # batch: (cerra_now, era_now, cerra_next)
        x_cerra, x_era, y = batch
        B = x_cerra.size(0)
        t_int = torch.ones(B, device=self.device) * 1.0

        y_hat = self(x_cerra, x_era, t_int)
        loss = self.loss_fn(y_hat, y)
        self.log(f"{stage}_loss", loss,
                 on_step=(stage=='train'),
                 on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, 'test')

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)


if __name__ == "__main__":
    #  Seed for Reproducibility
    seed_everything(42)

    root_dir = './data'
    variables = [
        "t_100000","t_92500","t_85000","t_70000","t_50000","t_30000","t_20000",
        "z_100000","z_92500","z_85000","z_70000","z_50000","z_30000","z_20000",
        "u_100000","u_92500","u_85000","u_70000","u_50000","u_30000","u_20000",
        "v_100000","v_92500","v_85000","v_70000","v_50000","v_30000","v_20000",
        "r_100000","r_92500","r_85000","r_70000","r_50000","r_30000","r_20000",
    ]
    batch_size         = 4
    test_batch_size    = 4
    img_size_cerra     = (16, 16)
    img_size_era       = (32, 32)
    patch_size_cerra   = 1
    patch_size_era     = 4
    embed_dim          = 256
    depth              = 5
    num_heads          = 4
    mlp_ratio          = 4.0
    lr                 = 5e-4
    max_epochs         = 50

    data_module = DLWCDataModule(
        root_dir        = root_dir,
        variables       = variables,
        batch_size      = batch_size,
        test_batch_size = test_batch_size
    )

    lit_model = WeatherForecast(
        variables        = variables,
        img_size_cerra   = img_size_cerra,
        img_size_era     = img_size_era,
        patch_size_cerra = patch_size_cerra,
        patch_size_era   = patch_size_era,
        embed_dim        = embed_dim,
        depth            = depth,
        num_heads        = num_heads,
        mlp_ratio        = mlp_ratio,
        lr               = lr,
    )

    logger = CSVLogger("lightning_logs")
    checkpoint_cb = ModelCheckpoint(
        monitor='val_loss', mode='min', save_top_k=3, save_last=True,
        filename='{epoch:02d}-{val_loss:.4f}'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = Trainer(
        logger            = logger,
        callbacks         = [checkpoint_cb, lr_monitor],
        max_epochs        = max_epochs,
        accelerator       = "gpu" if torch.cuda.is_available() else "cpu",
        devices           = 1 if torch.cuda.is_available() else None,
        precision         = "16-mixed" if torch.cuda.is_available() else 32,
        gradient_clip_val = 1.0,
        log_every_n_steps = 50,
    )

    trainer.fit(lit_model, datamodule=data_module)
    trainer.test(lit_model, datamodule=data_module)
