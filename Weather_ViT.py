import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, trunc_normal_
from weather_embedding import WeatherEmbeddingLite

class SimpleTimestepEmbedder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU()
        )

    def forward(self, t):
        return self.linear(t.unsqueeze(-1))  # (B, D)

class SimpleWeatherTransformer(nn.Module):
    def __init__(
        self,
        variables,
        img_size,
        patch_size=4,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
    ):
        super().__init__()
        # Embedding for spatial patches and variables
        self.embedding = WeatherEmbeddingLite(
            variables=variables,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        self.norm_embed = nn.LayerNorm(embed_dim)

        # Time embedding
        self.time_embed = SimpleTimestepEmbedder(embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        # Output head: map back to patch pixels
        self.head = nn.Linear(embed_dim, patch_size * patch_size)

        self.patch_size = patch_size
        self.variables = variables
        self.img_size = img_size

        self._init_weights()

    def _init_weights(self):
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def unpatchify(self, x):
        # x: (B, L, p*p)
        B, L, patch_area = x.shape
        p = self.patch_size
        V = len(self.variables)
        H, W = self.img_size
        Gh, Gw = H // p, W // p

        expected_L = Gh * Gw * V
        if L != expected_L:
            raise ValueError(f"Invalid number of patches. Got {L}, expected {expected_L} (Gh*Gw*V).")

        x = x.view(B, V, Gh, Gw, p, p)
        x = x.permute(0, 1, 2, 4, 3, 5)  # (B, V, Gh, p, Gw, p)
        x = x.contiguous().view(B, V, H, W)
        return x


    def forward(self, x, variables, time_interval):
        # x: (B, V, H, W)
        B = x.size(0)
        # Embed spatial+variable
        x_emb = self.embedding(x, variables)         # (B, L, D)
        x_emb = self.norm_embed(x_emb)

        # Add time embedding to each token
        t_emb = self.time_embed(time_interval)       # (B, D)
        x_emb = x_emb + t_emb.unsqueeze(1)

        # Transformer
        x_trans = self.transformer(x_emb)            # (B, L, D)

        # Head
        out = self.head(x_trans)                     # (B, L, V*p*p)
        # Reconstruct images
        out = self.unpatchify(out)
        return out

# Example usage:
# model = SimpleWeatherTransformer(
#     variables=['t', 'u', 'v'],
#     img_size=(64, 64),
#     patch_size=8,
# )
# device = 'cuda'
# x = torch.randn(2, 3, 64, 64).to(device)
# time_int = torch.tensor([1.0, 2.0]).to(device)
# y = model(x, ['t', 'u', 'v'], time_int)
