import jaxtyping
import math
import torch
import torch.nn as nn
from einops import einsum, rearrange


class Linear(torch.nn.Module):
  def __init__(
    self, in_features: int, out_features: int, device=None, dtype=None
  ):
    super().__init__()
    self.device = device
    self.weights = nn.Parameter(
      torch.empty(out_features, in_features, device=device, dtype=dtype)
    )
    mean = 0
    stddev = (2 / (in_features + out_features)) ** 0.5
    nn.init.trunc_normal_(
      self.weights, mean=mean, std=stddev, a=-3 * stddev, b=3 * stddev
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    ret = einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")
    return ret


class Embedding(nn.Module):
  def __init__(
    self, num_embeddings: int, embedding_dim: int, device=None, dtype=None
  ):
    super().__init__()
    self.weights = nn.Parameter(
      torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
    )
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

    rms = torch.sqrt(
      self.eps + einsum(x, x, "... d_model, ... d_model -> ...") / self.d_model
    )

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

    self.w1 = nn.Parameter(
      torch.ones(self.d_ff, d_model, device=device, dtype=dtype)
    )
    self.w2 = nn.Parameter(
      torch.ones(d_model, self.d_ff, device=device, dtype=dtype)
    )
    self.w3 = nn.Parameter(
      torch.ones(self.d_ff, d_model, device=device, dtype=dtype)
    )

  def forward(
    self, x: jaxtyping.Float[torch.Tensor, "... d_model"]
  ) -> jaxtyping.Float[torch.Tensor, "... d_model"]:
    a = silu(einsum(self.w1, x, "d_ff d_model, ... d_model -> ... d_ff"))
    b = einsum(self.w3, x, "d_ff d_model, ... d_model -> ... d_ff")
    y = einsum(a, b, "... d_ff, ... d_ff -> ... d_ff")
    y = einsum(self.w2, y, "d_model d_ff, ... d_ff -> ... d_model")
    return y


class RoPE(nn.Module):
  def __init__(self, d_k: int, max_seq_len: int = 2048, theta: float = 10000.0):
    """
    Args:
        d_k: Dimension of queries and keys (must be even).
        max_seq_len: Maximum expected sequence length.
        theta: The base value for frequency calculation.
    """
    super().__init__()
    assert d_k % 2 == 0, "RoPE dimension must be even."
    self.d_k = d_k

    # 1. Compute the frequencies for each consecutive pair: shape [d_k // 2]
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
    self.register_buffer("inv_freq", inv_freq, persistent=False)

    # 2. Compute sequence positions: shape [max_seq_len]
    t = torch.arange(max_seq_len, dtype=torch.float32)

    # 3. Outer product creates frequency matrix: shape [max_seq_len, d_k // 2]
    freqs = torch.einsum("i,j->ij", t, self.inv_freq)

    # Interleave frequencies so they match consecutive items: [f0, f0, f1, f1, ...]
    # This makes sure index 0 and index 1 share the exact same frequency!
    emb = torch.stack((freqs, freqs), dim=-1).flatten(-2)

    # Precompute and cache cos and sin layouts: shape [max_seq_len, d_k]
    self.register_buffer("cos_cached", emb.cos(), persistent=False)
    self.register_buffer("sin_cached", emb.sin(), persistent=False)

  def _rotate_consecutive(self, x: torch.Tensor) -> torch.Tensor:
    """
    Rotates consecutive pairs: transforms [x0, x1, x2, x3] into [-x1, x0, -x3, x2]
    """
    # Group into pairs along the last dimension
    x_paired = x.view(*x.shape[:-1], self.d_k // 2, 2)
    x1, x2 = x_paired.unbind(dim=-1)

    # Apply rotation swap (-x2, x1)
    rotated_paired = torch.stack((-x2, x1), dim=-1)
    return rotated_paired.view(*x.shape)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: Input tensor of shape [... sequence_length d_k]
    """
    seq_len = x.shape[-2]

    # Slice cache down to target length
    cos = self.cos_cached[:seq_len, :]
    sin = self.sin_cached[:seq_len, :]

    # Build dynamic broadcasting shapes to align perfectly on the left
    # (e.g., shifts [12, 64] -> [1, 12, 64] to seamlessly match the input batch)
    view_shape = [1] * (x.ndim - 2) + [seq_len, self.d_k]
    cos = cos.view(view_shape)
    sin = sin.view(view_shape)

    # Standard RoPE formula applied on consecutive elements
    return (x * cos) + (self._rotate_consecutive(x) * sin)


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
  """softmax.

  Args:
    x: the tensor
    i: the i-th dimension to apply the softmax.

  Return:
    the tensor after softmax
  """
  m = x.max(dim=i, keepdim=True).values
  x = x - m
  x_exp = x.exp()
  ret = x_exp / x_exp.sum(dim=i, keepdim=True)
  return ret


def scaled_dot_product_attention(
  q: jaxtyping.Float[torch.Tensor, '... queries d_k'],
  k: jaxtyping.Float[torch.Tensor, '... keys d_k'],
  v: jaxtyping.Float[torch.Tensor, '... keys d_v'],
  mask: jaxtyping.Float[torch.Tensor, '... queries keys'] | None = None,
) -> torch.Tensor:
  print(f"===lizhi {k.shape=} {q.shape=} {v.shape=}")
  d_k = k.shape[-1]
  qk = einsum(q, k, "... queries d_k, ... keys d_k -> ... queries keys")
  qk_normalized = qk / math.sqrt(d_k)
  print(f"===lizhi {qk_normalized.shape=}")
  if mask is not None:
    print(f"===lizhi {mask.shape=}")
    qk_normalized.masked_fill_(~mask, float('-inf'))
  s = softmax(qk_normalized, i=-1)
  ret = einsum(s, v, "... s_0 s, ... s d_v -> ... s_0 d_v")
  return ret
