import math
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from weather_embedding_lite import WeatherEmbeddingLite


class WeatherViTSeq2Seq(pl.LightningModule):
    def __init__(
        self,
        variables: List[str],
        img_size=(16, 16),
        patch_size=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        lr=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.variables = variables
        self.num_vars = len(variables)
        self.patch_size = patch_size

        # Embedding for encoder (no separate decoder embed needed)
        self.encoder_embed = WeatherEmbeddingLite(
            variables=variables,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=depth)

        # Output projection from token embeddings to patch pixels
        self.output_proj = nn.Linear(embed_dim, patch_size * patch_size)

        # Compute patch grid resolution
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.patch_res = int(math.sqrt(num_patches))

        # Final conv to assemble patches back to multi-channel image
        # takes single reconstructed channel and maps to one channel per variable
        self.final_conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.num_vars,
            kernel_size=1
        )

        self.loss_fn = nn.MSELoss()
        self.lr = lr

    def forward(self, src_img: torch.Tensor) -> torch.Tensor:
        # src_img: [B, V, H, W]
        # 1. Patch embedding
        src_tok = self.encoder_embed(src_img, self.variables)  # [B, N, D]
        # 2. Encode
        memory = self.encoder(src_tok)                         # [B, N, D]
        # 3. Decode: use encoder tokens as decoder input
        out_tok = self.decoder(src_tok, memory)                # [B, N, D]

        # 4. Project tokens to patch pixels
        patches = self.output_proj(out_tok)  # [B, N, P*P]
        B, N, _ = patches.shape

        # 5. Reshape to (B, N, 1, p, p)
        patches = patches.view(B, N, 1, self.patch_size, self.patch_size)
        # 6. Rearrange into image grid
        patches = patches.permute(0, 2, 1, 3, 4)
        patches = patches.reshape(
            B,
            1,
            self.patch_res,
            self.patch_res,
            self.patch_size,
            self.patch_size
        )
        patches = patches.permute(0, 1, 2, 4, 3, 5)
        recon = patches.reshape(
            B,
            1,
            self.patch_res * self.patch_size,
            self.patch_res * self.patch_size
        )

        # 7. Final conv to map to V output channels
        return self.final_conv(recon)  # [B, V, H, W]

    def training_step(self, batch, batch_idx):
        src, tgt = batch
        pred = self(src)
        loss = self.loss_fn(pred, tgt)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        pred = self(src)
        loss = self.loss_fn(pred, tgt)
        self.log("val/loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
