import jaxtyping
import math
import os
import torch
import torch.nn as nn
import numpy as np
import typing
from einops import einsum, rearrange


def save_checkpoint(
  model: nn.Module,
  optimizer: torch.optim.Optimizer,
  iteration: int,
  out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
  state = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "iteration": iteration,
  }
  torch.save(state, out)


def load_checkpoint(
  src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
  model: nn.Module,
  optimizer: torch.optim.Optimizer,
) -> int:
  state = torch.load(src)
  model.load_state_dict(state["model"])
  optimizer.load_state_dict(state["optimizer"])
  return state["iteration"]
