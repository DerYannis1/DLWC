Input: [B, V, H, W]
        │
WeatherEmbeddingLite (Patch + Pos + Var Embedding)
        ↓
Tensor: [B, V*L, D]
        ↓
Transformer Encoder (12x Block)
        ↓
Tensor: [B, V*L, D]
        ↓
Umformen in Patch-Grid: [B, D, H', W']
        ↓
Decoder CNN
        ↓
Output: [B, out_channels, H, W]