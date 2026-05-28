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


def learning_rate_schedule(
  t: int, a_max: float, a_min: float, tw: int, tc: int
) -> float:
  """Cosine annealing learning rate schedule.

  Args:
    t: the current iteration.
    a_max: the max learning rate
    a_min: the min (final) learning rate
    tw: the # of warm-up iterations
    tc: the final iteration of cosine annealing. (max # of iterations)
  """
  if t < tw:
    return t / tw * a_max
  elif t <= tc:
    return a_min + 0.5 * (1 + math.cos((t - tw) / (tc - tw) * math.pi)) * (
      a_max - a_min
    )
  else:
    return a_min


def run_gradient_clipping(
  parameters: typing.Iterable[torch.nn.Parameter],
  max_l2_norm: float,
  eps: float = 1e-6,
) -> None:
  # g_l2_norm is the l2_norm across all parameters' gradients.
 
  square_sum = 0.0
  param_list = [p for p in parameters if p.grad is not None]
  for p in param_list:
    square_sum += torch.sum(torch.pow(p.grad, 2))

  g_l2_norm = math.sqrt(square_sum)

  if g_l2_norm > max_l2_norm:
    for p in param_list:
      # In-place update
      p.grad.mul_(max_l2_norm / (g_l2_norm + eps))
