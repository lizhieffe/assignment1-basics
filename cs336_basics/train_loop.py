# Run:
#   uv pip install -e . \
#   uv run cs336_basics/train_loop.py

from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import datetime
from typing import IO, Any, BinaryIO

import numpy as np
import numpy.typing as npt
import torch
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAINING_QUERYSET = "data/TinyStoriesV2-GPT4-train.tokens.bin"


def main():
  uuid = datetime.now().strftime("%Y%m%d%H%M%S")

  tokenizer = AutoTokenizer.from_pretrained(constants.TOKENIZER_NAME)
  vocab_size = tokenizer.vocab_size
  dtype = np.uint16 if vocab_size < 65536 else np.uint32

  print(f"{vocab_size=} {dtype=} {DEVICE=}")

  transformer_lm = model.TransformerLM(
    vocab_size=vocab_size,
    context_length=CONTEXT_LENGTH,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_model=D_MODEL,
    d_ff=None,
    rope_theta=10000,
    device=DEVICE
  )

  optz = optimizer.AdamW(
    params=transformer_lm.parameters(),
    lr=1e-3,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
  )

  big_array = np.memmap(TRAINING_QUERYSET, dtype=dtype, mode="r")

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

  it = 1
  for _ in range(NUM_ITERS):
    print(f"============== it = {it} ====================")

    optz.zero_grad()
    x, y = training.sample_training_data(
      x=big_array,
      batch_size=BATCH_SIZE,
      context_length=CONTEXT_LENGTH,
      device=DEVICE,
    )
    print(f"Sampled training data {x.shape=} {y.shape=}")
    logits = transformer_lm(x)
    loss = loss_lib.cross_entropy(logits=logits, targets=y)
    print(f"Loss = {loss.item():.3f}")
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

    it += 1


if __name__ == "__main__":
  main()
