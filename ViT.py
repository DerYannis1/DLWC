# vit_model.py

import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
from timm import create_model

class WeatherViT(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,    # e.g. 35 (5 variables × 7 levels)
        out_channels: int,   # same as in_channels
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ViT for 16×16 → 4×4 patches
        self.vit = create_model(
            "vit_base_patch4_16",
            pretrained=False,
            in_chans=in_channels,
            num_classes=0,     # drop the head
        )

        # compute patch grid size
        num_patches = (16 // 4) ** 2  # 16
        embed_dim   = self.vit.embed_dim
        self.patch_res = int(math.sqrt(num_patches))  # 4

        # a tiny decoder: 4×4 → 16×16
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(
                embed_dim,
                embed_dim // 2,
                kernel_size=4,
                stride=4
            ),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, out_channels, kernel_size=1),
        )

        self.loss_fn = nn.MSELoss()
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, 16, 16]
        feats = self.vit.forward_features(x)
        # feats: [B, num_patches+1, embed_dim]
        tokens = feats[:, 1:, :]  # drop CLS token → [B,16,embed_dim]
        B, N, D = tokens.shape
        h   = self.patch_res  # 4
        tokens = tokens.transpose(1, 2).reshape(B, D, h, h)
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
