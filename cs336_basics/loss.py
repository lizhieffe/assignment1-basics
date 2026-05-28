import jaxtyping
import math
import torch
import torch.nn as nn
from einops import einsum, rearrange

from cs336_basics import model


def cross_entropy(
  logits: jaxtyping.Float[torch.Tensor, "... vocab_size"],
  targets: jaxtyping.Int[torch.Tensor, "..."],
) -> jaxtyping.Float[torch.Tensor, ""]:
  # Support both 2D (b vocab_size) and 3D (b s vocab_size) logits. 
  logits = rearrange(logits, '... vocab_size -> (...) vocab_size')

  # Support both 1D (b) and 2D (b s) targets.
  targets = rearrange(targets, "... -> (...)")
  
  D = logits.shape[0]

  # We don't use softmax, and instead cancel out the log in NLL and exp in norminator of softmax.
  logits_max = logits.max(dim=-1, keepdim=True).values
  logits_normalized = logits - logits_max
  log_logits_normalized = logits_normalized - torch.log(
    torch.sum(torch.exp(logits_normalized), dim=-1, keepdim=True)
  )
  return -1 * torch.mean(log_logits_normalized[torch.arange(D), targets])
