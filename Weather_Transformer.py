import torch
import torch.nn as nn
from timm.models.vision_transformer import trunc_normal_
from weather_embedding import WeatherEmbedding

class SimpleTimestepEmbedder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU()
        )

    def forward(self, t):
        return self.linear(t.unsqueeze(-1))  # (B, D)

class DLWCTransformer(nn.Module):
    def __init__(
        self,
        variables,
        img_size,
        img_size_era,
        patch_size=4,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.variables     = variables
        self.img_size      = img_size         #cerra 16x16
        self.img_size_era  = img_size_era
        self.patch_size    = patch_size

        # Zwei Embeddings: eins für CERRA (16x16) und eins für ERA (32x32)
        self.embed_cerra = WeatherEmbedding(
            variables=variables,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        self.embed_era   = WeatherEmbedding(
            variables=variables,
            img_size=img_size_era,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        self.norm_embed = nn.LayerNorm(embed_dim)
        self.time_embed = SimpleTimestepEmbedder(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Head: projiziert jeden Token zu p*p Pixeln
        self.head = nn.Linear(embed_dim, patch_size * patch_size)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def unpatchify(self, small_tokens):
        # small_tokens: (B, L_small, p*p)
        B, L_small, pp = small_tokens.shape
        p = self.patch_size
        V = len(self.variables)
        H, W = self.img_size
        Gh, Gw = H // p, W // p
        expected = V * Gh * Gw
        if L_small != expected:
            raise ValueError(f"Expected {expected} small tokens, got {L_small}")
        x = small_tokens.view(B, V, Gh, Gw, p, p)
        x = x.permute(0,1,2,4,3,5).contiguous().view(B, V, H, W)
        return x

    def forward(self, x_cerra, x_era, time_interval):
        """
        x_cerra: (B, V, H, W)
        x_era:   (B, V, H_era, W_era)
        time_interval: (B,) or (B,1)
        """
        emb_c = self.embed_cerra(x_cerra, self.variables)  # (B, L_small, D)
        emb_e = self.embed_era  (x_era,   self.variables)  # (B, L_era,   D)
        emb_c = self.norm_embed(emb_c)
        emb_e = self.norm_embed(emb_e)

        t_emb = self.time_embed(time_interval)             # (B, D)
        emb_c = emb_c + t_emb.unsqueeze(1)
        emb_e = emb_e + t_emb.unsqueeze(1)

        # 3) Tokens zusammenführen
        tokens = torch.cat([emb_c, emb_e], dim=1)          # (B, L_small+L_era, D)

        x_trans = self.transformer(tokens)                 # (B, L_total, D)

        patches = self.head(x_trans)                       # (B, L_total, p*p)

        L_small = emb_c.shape[1]
        small_patches = patches[:, :L_small, :]           # (B, L_small, p*p)
        out = self.unpatchify(small_patches)               # (B, V, H, W)
        return out
