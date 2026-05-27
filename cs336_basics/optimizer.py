import jaxtyping
import math
import torch
import torch.nn as nn
import typing
from einops import einsum, rearrange

from cs336_basics import model


class AdamW(torch.optim.Optimizer):
  def __init__(
    self,
    params,
    betas: tuple[float, float],
    weight_decay: float,
    lr=1e-3,
    eps: float = 1e-8,
  ):
    """
    Args:
      params: params. (before the other hyperparameters)
      betas: 1st and 2nd momentum rate.
      weight_decay: weight decay rate.
      lr: learning rate. A.k.a alpha.
      eps: numerical stability.
    """
    if lr < 0:
      raise ValueError(f"Invalid learning rate: {lr}")
    defaults = {
      "lr": lr,
      "beta1": betas[0],
      "beta2": betas[1],
      "lamda": weight_decay,
      "eps": eps,
    }
    super().__init__(params, defaults=defaults)

  def step(self, closure: typing.Callable | None = None):
    """
    Args:
      closure: recompute the loss.
    """
    loss = None if closure is None else closure()

    for group in self.param_groups:
      lr = group["lr"]
      beta1 = group["beta1"]
      beta2 = group["beta2"]
      lamda = group["lamda"]
      eps = group["eps"]

      for p in group["params"]:
        if p.grad is None:
          continue

        state = self.state[p]
        t = state.get("t", 1)
        m = state.get("m", torch.zeros_like(p.data))
        v = state.get("v", torch.zeros_like(p.data))

        grad = p.grad.data
        # print(f"===lizhi {t=} {beta1=}")
        lr_t = lr * math.sqrt(1 - math.pow(beta2, t)) / (1 - math.pow(beta1, t))
        p.data -= lr * lamda * p.data
        new_m = beta1 * m + (1 - beta1) * grad
        new_v = beta2 * v + (1 - beta2) * grad**2
        p.data -= lr_t * new_m / torch.sqrt(new_v) + eps

        state["m"] = new_m
        state["v"] = new_v
        state["t"] = t + 1

    return loss
