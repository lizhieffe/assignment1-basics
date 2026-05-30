# Run:
#   uv pip install -e . \
#   uv run cs336_basics/train_loop.py
#
# Training metrics: https://wandb.ai/lizhieffe-teestory/cs336-assignment-1

from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import datetime
from typing import IO, Any, BinaryIO

import jaxtyping
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import wandb
from jaxtyping import Bool, Float, Int
from torch import Tensor
from transformers import AutoTokenizer

from cs336_basics import checkpoint, constants, model, optimizer, training
from cs336_basics import loss as loss_lib


NUM_LAYERS = 2
NUM_HEADS = 4
D_MODEL = 128
BATCH_SIZE = 32
CONTEXT_LENGTH = 32
SEQUENCE_LENGTH = 20


NUM_ITERS = 1000
SAVE_CKPT_EVERY_N_ITERS = 200

EVAL_QUERY_COUNT = 1024
EVAL_EVERY_N_ITERS = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAINING_QUERYSET = "data/TinyStoriesV2-GPT4-train.tokens.bin"
EVAL_QUERYSET = "data/TinyStoriesV2-GPT4-valid.tokens.bin"


def run_eval(
  x: jaxtyping.Int(torch.Tensor, "b s"),
  targets: jaxtyping.Int(torch.Tensor, "b s"),
  model: nn.Module,
  BATCH_SIZE: int = 32,  # Added BATCH_SIZE parameter
) -> float:
  """Run eval in batches to avoid OOM, returning the exact overall mean loss.

  Returns:
    loss
  """
  # Split x and targets along the batch dimension (dim=0)
  # torch.split handles cases where the total size isn't perfectly divisible
  x_batches = torch.split(x, BATCH_SIZE, dim=0)
  targets_batches = torch.split(targets, BATCH_SIZE, dim=0)

  total_loss_sum = 0.0
  total_samples = 0

  # Put model in eval mode if it isn't already
  model.eval()

  # Disable gradient tracking for evaluation speed and memory efficiency
  with torch.no_grad():
    for x_batch, target_batch in zip(x_batches, targets_batches):
      current_batch_size = x_batch.shape[0]

      # Forward pass on the individual batch
      logits = model(x_batch)
      loss = loss_lib.cross_entropy(logits=logits, targets=target_batch)

      # Weight the loss by current batch size to handle uneven final batches
      total_loss_sum += loss.item() * current_batch_size
      total_samples += current_batch_size

  # Return the true mathematically weighted average loss
  return total_loss_sum / total_samples


def main():
  uuid = datetime.now().strftime("%Y%m%d%H%M%S")

  tokenizer = AutoTokenizer.from_pretrained(constants.TOKENIZER_NAME)
  vocab_size = tokenizer.vocab_size
  dtype = np.uint16 if vocab_size < 65536 else np.uint32

  config = {
    "num_layers": NUM_LAYERS,
    "num_heads": NUM_HEADS,
    "d_model": D_MODEL,
    "batch_size": BATCH_SIZE,
    "context_length": CONTEXT_LENGTH,
    "num_iters": NUM_ITERS,
    "lr": 1e-3,
    "weight_decay": 0.01,
    "device": DEVICE,
  }
  print(f"Start training with config: {config}")
  wandb.init(project="cs336-assignment-1", name=f"run-{uuid}", config=config)

  transformer_lm = model.TransformerLM(
    vocab_size=vocab_size,
    context_length=CONTEXT_LENGTH,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_model=D_MODEL,
    d_ff=None,
    rope_theta=10000,
    device=DEVICE,
  )

  # Track gradients and model topology.
  wandb.watch(transformer_lm, log="gradients", log_freq=EVAL_EVERY_N_ITERS)

  optz = optimizer.AdamW(
    params=transformer_lm.parameters(),
    lr=1e-3,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
  )

  big_array = np.memmap(TRAINING_QUERYSET, dtype=dtype, mode="r")
  eval_query_array = np.memmap(EVAL_QUERYSET, dtype=dtype, mode="r")

  out_dir = f"experiments/{uuid}/ckpts"
  os.makedirs(out_dir, exist_ok=False)
  print(f"Verified checkpoint directory structure exists at: {out_dir}")

  if SAVE_CKPT_EVERY_N_ITERS and SAVE_CKPT_EVERY_N_ITERS > 0:
    ckpt_dir = f"{out_dir}/{str(0).zfill(6)}"
    os.makedirs(ckpt_dir, exist_ok=False)
    ckpt_file = f"{ckpt_dir}/ckpt"
    checkpoint.save_checkpoint(
      model=transformer_lm,
      optimizer=optz,
      iteration=0,
      out=ckpt_file,
    )
    print(f"Saved initial ckpt to {ckpt_file}")

  # Have a fixed eval set.
  eval_x, eval_targets = training.sample_training_data(
    x=eval_query_array,
    batch_size=EVAL_QUERY_COUNT,
    context_length=CONTEXT_LENGTH,
    device=DEVICE,
  )

  eval_loss = run_eval(x=eval_x, targets=eval_targets, model=transformer_lm)
  print(f"Initial CKPT Eval Loss = {eval_loss:.3f}")
  wandb.log({"eval/loss": eval_loss}, step=0)

  it = 1
  for _ in range(NUM_ITERS):
    optz.zero_grad()
    x, y = training.sample_training_data(
      x=big_array,
      batch_size=BATCH_SIZE,
      context_length=CONTEXT_LENGTH,
      device=DEVICE,
    )
    logits = transformer_lm(x)
    loss = loss_lib.cross_entropy(logits=logits, targets=y)
    if it % EVAL_EVERY_N_ITERS == 0:
      print(f"============== it = {it} ====================")
      loss_val = loss.item()
      print(f"Train Loss = {loss_val:.3f}")
      wandb.log({"train/loss": loss_val}, step=it)
    loss.backward()
    optz.step()

    if (
      SAVE_CKPT_EVERY_N_ITERS
      and SAVE_CKPT_EVERY_N_ITERS > 0
      and it % SAVE_CKPT_EVERY_N_ITERS == 0
    ):
      ckpt_dir = f"{out_dir}/{str(it).zfill(6)}"
      os.makedirs(ckpt_dir, exist_ok=False)
      ckpt_file = f"{ckpt_dir}/ckpt"
      checkpoint.save_checkpoint(
        model=transformer_lm,
        optimizer=optz,
        iteration=it,
        out=ckpt_file,
      )
      print(f"Saved ckpt to {ckpt_file}")

    if it % EVAL_EVERY_N_ITERS == 0:
      eval_loss = run_eval(x=eval_x, targets=eval_targets, model=transformer_lm)
      print(f"Eval Loss = {eval_loss:.3f}")
      wandb.log({"eval/loss": eval_loss}, step=it)

    it += 1

  wandb.finish()


if __name__ == "__main__":
  main()
