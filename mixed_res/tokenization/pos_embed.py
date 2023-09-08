import math

import torch
from torch import nn


def build_sinusoidal_embeddings(positions: torch.Tensor,
                                embedding_dim: int) -> torch.FloatTensor:
    """
    Based on the implementation in fairseq:
    https://github.com/facebookresearch/fairseq/blob/5ecbbf58d6e80b917340bcbf9d7bdbb539f0f92b/fairseq/modules/sinusoidal_positional_embedding.py#L36
    """
    assert positions.ndim == 2  # [batch, position]
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float, device=positions.device) *
        -emb)
    emb = positions.unsqueeze(-1) * emb.view(1, 1, -1)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        # zero pad
        emb = torch.cat([emb, torch.zeros_like(emb[:, :, :1])], dim=-1)
    return emb


class SinusoidalEmbeddings(nn.Module):

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, positions: torch.Tensor) -> torch.FloatTensor:
        """
        positions: [batch, position]
                   Can be integers or floats (e.g. position 4.5)
        """
        return build_sinusoidal_embeddings(positions, self.embedding_dim)
