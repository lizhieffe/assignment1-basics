import jaxtyping
import math
import torch
import torch.nn as nn
from einops import einsum, rearrange

from cs336_basics import model


def cross_entropy(
  logits: jaxtyping.Float[torch.Tensor, "b vocab_size"],
  targets: jaxtyping.Int[torch.Tensor, "b"],
) -> jaxtyping.Float[torch.Tensor, ""]:
  D = logits.shape[0]

  # We don't use softmax, and instead cancel out the log in NLL and exp in norminator of softmax.
  logits_max = logits.max(dim=-1, keepdim=True).values
  logits_normalized = logits - logits_max
  log_logits_normalized = logits_normalized - torch.log(
    torch.sum(torch.exp(logits_normalized), dim=-1, keepdim=True)
  )
  return -1 * torch.mean(log_logits_normalized[torch.arange(D), targets])
