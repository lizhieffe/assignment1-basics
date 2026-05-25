import jaxtyping
import math
import torch
import torch.nn as nn
from einops import einsum, rearrange


class Linear(torch.nn.Module):
  def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
    super().__init__()
    self.device = device
    self.weights = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
    mean = 0
    stddev = (2 / (in_features + out_features)) ** 0.5
    nn.init.trunc_normal_(self.weights, mean=mean, std=stddev, a=-3 * stddev, b=3 * stddev)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    ret = einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")
    return ret


class Embedding(nn.Module):
  def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
    super().__init__()
    self.weights = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
    nn.init.trunc_normal_(self.weights, mean=0, std=1, a=-3, b=3)

  def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
    return self.weights[token_ids]


class RMSNorm(nn.Module):
  def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
    super().__init__()
    self.eps = eps
    self.d_model = d_model
    self.weights = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    in_type = x.dtype
    x = x.to(torch.float32)

    rms = torch.sqrt(self.eps + einsum(x, x, "... d_model, ... d_model -> ...") / self.d_model)

    y = einsum(x, self.weights, "... d_model, d_model -> ... d_model")

    y = y / rearrange(rms, "... -> ... 1")
    y = y.to(in_type)
    return y


def silu(x: torch.Tensor) -> torch.Tensor:
  return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
  def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
    """SwiGLU

    Args:
      d_model: the input/output dim.
      d_ff: the hidden dim. If it is not given, it is calculated as roughly 8/3 of the d_model but ceil to multiplier of 64.
      device: device.
      dtype: dtype
    """
    super().__init__()

    if d_ff:
      self.d_ff = d_ff
    else:
      self.d_ff = math.ceil(8 * d_model / 3.0) * 64

    self.w1 = nn.Parameter(torch.ones(self.d_ff, d_model, device=device, dtype=dtype))
    self.w2 = nn.Parameter(torch.ones(d_model, self.d_ff, device=device, dtype=dtype))
    self.w3 = nn.Parameter(torch.ones(self.d_ff, d_model, device=device, dtype=dtype))

  def forward(self, x: jaxtyping.Float[torch.Tensor, '... d_model']) -> torch.Tensor:
    a = silu(einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff"))
    b = einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")
    y = einsum(a, b, "... d_ff, ... d_ff -> ... d_ff")
    y = einsum(self.w2, y, "d_model d_ff, ... d_ff -> ... d_model")
    return y
