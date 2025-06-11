import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, trunc_normal_

from pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid

class WeatherEmbeddingLite(nn.Module):
    def __init__(
        self,
        variables,
        img_size,
        patch_size=4,
        embed_dim=512,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.variables = variables

        self.token_embeds = nn.ModuleList([
            PatchEmbed(None, patch_size, 1, embed_dim) for _ in range(len(variables))
        ])
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        self.channel_embed, self.channel_map = self.create_var_embedding(embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)

        self.initialize_weights()

    def initialize_weights(self):
        # 2D Positional Encoding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            self.img_size[0] // self.patch_size,
            self.img_size[1] // self.patch_size,
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # 1D Variable Embedding
        channel_embed = get_1d_sincos_pos_embed_from_grid(
            self.channel_embed.shape[-1],
            np.arange(len(self.variables)),
        )
        self.channel_embed.data.copy_(torch.from_numpy(channel_embed).float().unsqueeze(0))

        # PatchEmbed init
        for token_embed in self.token_embeds:
            w = token_embed.proj.weight.data
            trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

    def create_var_embedding(self, dim):
        var_embed = nn.Parameter(torch.zeros(1, len(self.variables), dim), requires_grad=True)
        var_map = {var: idx for idx, var in enumerate(self.variables)}
        return var_embed, var_map

    def get_var_ids(self, vars, device):
        ids = np.array([self.channel_map[var] for var in vars])
        return torch.from_numpy(ids).to(device)

    def get_var_emb(self, var_emb, vars):
        ids = self.get_var_ids(vars, var_emb.device)
        return var_emb[:, ids, :]

    def forward(self, x: torch.Tensor, variables):
        if isinstance(variables, list):
            variables = tuple(variables)

        embeds = []
        var_ids = self.get_var_ids(variables, x.device)

        for i in range(len(var_ids)):
            id = var_ids[i]
            embed_variable = self.token_embeds[id](x[:, i:i+1])  # B, L, D
            embeds.append(embed_variable)

        x = torch.stack(embeds, dim=1)  # B, V, L, D

        # Add variable and positional embeddings
        var_embed = self.get_var_emb(self.channel_embed, variables).unsqueeze(2)  # B, V, 1, D
        x = x + var_embed  # B, V, L, D
        x = x + self.pos_embed.unsqueeze(1)  # B, V, L, D

        # Flatten: combine variable and patch dimensions
        x = x.flatten(1, 2)  # B, V*L, D

        return x
