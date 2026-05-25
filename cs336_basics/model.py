import torch
import torch.nn as nn
from einops import einsum

class Linear(torch.nn.Module):
  def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
    super().__init__()
    self.device = device
    self.weights = nn.Parameter(torch.empty(out_features, in_features, device=device,dtype=dtype))
    mean = 0
    stddev = (2 / (in_features + out_features)) ** 0.5
    nn.init.trunc_normal_(self.weights, mean=mean, std=stddev, a=-3*stddev, b=3*stddev)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    ret = einsum(x, self.weights, '... d_in, d_out d_in -> ... d_out')
    return ret

class Embedding(nn.Module):
  def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
    super().__init__()
    self.weights = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
    nn.init.trunc_normal_(self.weights, mean=0, std=1, a=-3, b=3)

  def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
    return self.weights[token_ids]