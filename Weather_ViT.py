import math
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from weather_embedding import WeatherEmbeddingLite


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

        self.patch_res_h = img_size[0] // patch_size
        self.patch_res_w = img_size[1] // patch_size

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
        B, V, H, W = src_img.shape

        # 1. Patch embedding → [B, N, D], where N = V * (H/ps) * (W/ps)
        src_tok = self.encoder_embed(src_img, self.variables)

        # 2. Encode
        memory = self.encoder(src_tok)

        # 3. Decode (teacher-forcing)
        out_tok = self.decoder(src_tok, memory)  # [B, N, D]

        # 4. Project each token to a p×p patch → [B, N, p*p]
        patches = self.output_proj(out_tok)
        _, N, _ = patches.shape

        # 5. Reshape into per-variable, per-spatial-patch blocks:
        #    N = V * patch_res_h * patch_res_w
        Gh, Gw = self.patch_res_h, self.patch_res_w
        assert V * Gh * Gw == N, f"N={N} != V({V})*{Gh}*{Gw}"

        #   → [B, V, Gh, Gw, p, p]
        patches = patches.view(
            B,
            V,
            Gh,
            Gw,
            self.patch_size,
            self.patch_size
        )

        # 6. Permute so we can flatten patches into full images:
        #    [B, V, Gh, p, Gw, p]
        patches = patches.permute(0, 1, 2, 4, 3, 5)

        # 7. Collapse (Gh × p) and (Gw × p) to get [B, V, H, W]
        recon = patches.reshape(
            B,
            V,
            Gh * self.patch_size,
            Gw * self.patch_size
        )

        return recon

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
