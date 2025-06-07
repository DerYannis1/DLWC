from timm.models.vision_transformer import VisionTransformer
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
import torch
import torch.nn as nn
import math
from typing import Optional, List
from dataloader.dataloader import DLWCDataModule

class WeatherViT(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Use direct instantiation to allow 4x4 patches
        self.vit = VisionTransformer(
            img_size=16,
            patch_size=4,
            in_chans=in_channels,
            num_classes=0,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=torch.nn.LayerNorm,
        )

        num_patches = (16 // 4) ** 2  # = 16
        embed_dim = self.vit.embed_dim
        self.patch_res = int(math.sqrt(num_patches))  # = 4

        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=4, stride=4),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, out_channels, kernel_size=1),
        )

        self.loss_fn = nn.MSELoss()
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.vit.forward_features(x)  # [B, 17, D]
        tokens = feats[:, 1:, :]  # remove CLS → [B, 16, D]
        B, N, D = tokens.shape
        tokens = tokens.transpose(1, 2).reshape(B, D, self.patch_res, self.patch_res)
        return self.decoder(tokens)

    def training_step(self, batch, batch_idx):
        inp, target, _ = batch
        pred = self(inp)
        loss = self.loss_fn(pred, target)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inp, target, _ = batch
        pred = self(inp)
        loss = self.loss_fn(pred, target)
        self.log("val/loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
