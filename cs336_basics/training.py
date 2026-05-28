import jaxtyping
import math
import torch
import torch.nn as nn
import numpy as np
import typing
from einops import einsum, rearrange

from cs336_basics import model


def sample_training_data(
  x: np.ndarray, batch_size: int, context_length: int, device: str
) -> tuple[
  jaxtyping.Int[torch.Tensor, "batch_size context_length"],
  jaxtyping.Int[torch.Tensor, "batch_size context_length"],
]:
  # Setup your dummy variables for context
  high_bound = len(x) - context_length

  # 1. Initialize the modern random generator
  rng = np.random.default_rng()

  # 2. Sample unique indices
  sampled_ints = rng.choice(high_bound, size=batch_size, replace=False)

  inp = torch.Tensor(
    [x[start : start + context_length] for start in sampled_ints], device=device
  )
  target = torch.Tensor(
    [x[start + 1 : start + context_length + 1] for start in sampled_ints],
    device=device,
  )
  return (inp, target)
