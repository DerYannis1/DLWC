import torch
import torch.nn as nn
from timm.models.vision_transformer import trunc_normal_
from utils.weather_embedding import WeatherEmbedding

class TimestepEmbedder(nn.Module):
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
        img_size_cerra,
        img_size_era,
        patch_size_cerra=1,
        patch_size_era=4,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.variables        = variables
        self.img_size_cerra   = img_size_cerra
        self.img_size_era     = img_size_era
        self.patch_size_cerra = patch_size_cerra
        self.patch_size_era   = patch_size_era

        # Zwei Embeddings: eins für CERRA (small patches) und eins für ERA (larger patches)
        self.embed_cerra = WeatherEmbedding(
            variables=variables,
            img_size=img_size_cerra,
            patch_size=patch_size_cerra,
            embed_dim=embed_dim,
        )
        self.embed_era = WeatherEmbedding(
            variables=variables,
            img_size=img_size_era,
            patch_size=patch_size_era,
            embed_dim=embed_dim,
        )

        self.norm_embed = nn.LayerNorm(embed_dim)
        self.time_embed = TimestepEmbedder(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.head_cerra = nn.Linear(embed_dim, patch_size_cerra * patch_size_cerra)
        self.head_era   = nn.Linear(embed_dim, patch_size_era   * patch_size_era)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def unpatchify(self, small_tokens):
        B, L_small, pp = small_tokens.shape
        p = self.patch_size_cerra
        V = len(self.variables)
        H, W = self.img_size_cerra
        Gh, Gw = H // p, W // p
        expected = V * Gh * Gw
        #Debugging
        if L_small != expected:
            raise ValueError(f"Expected {expected} small tokens, got {L_small}")
        x = small_tokens.view(B, V, Gh, Gw, p, p)
        x = x.permute(0,1,2,4,3,5).contiguous().view(B, V, H, W)
        return x

    def forward(self, x_cerra, x_era, time_interval):
        """
        x_cerra: (B, V, H_c, W_c)
        x_era:   (B, V, H_e, W_e)
        time_interval: (B,) or (B,1)
        """
        emb_c = self.embed_cerra(x_cerra, self.variables)  # (B, L_c, D)
        emb_e = self.embed_era  (x_era,   self.variables)  # (B, L_e, D)
        emb_c = self.norm_embed(emb_c)
        emb_e = self.norm_embed(emb_e)

        t_emb = self.time_embed(time_interval)             # (B, D)
        emb_c = emb_c + t_emb.unsqueeze(1)
        emb_e = emb_e + t_emb.unsqueeze(1)

        tokens = torch.cat([emb_c, emb_e], dim=1)          # (B, L_c+L_e, D)

        x_trans = self.transformer(tokens)                 # (B, L_total, D)

        L_c = emb_c.shape[1]
        patches_c = self.head_cerra(x_trans[:, :L_c, :])   # (B, L_c, p_c^2)

        out = self.unpatchify(patches_c)                   # (B, V, H_c, W_c)
        return out
