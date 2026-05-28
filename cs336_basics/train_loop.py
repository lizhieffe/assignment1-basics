# Run:
#   uv pip install -e . \
#   uv run cs336_basics/train_loop.py

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from transformers import AutoTokenizer

from cs336_basics import checkpoint, model, optimizer, training
from cs336_basics import loss as loss_lib

NUM_LAYERS = 2
NUM_HEADS = 4
D_MODEL = 128

NUM_ITERS = 100

BATCH_SIZE = 32
CONTEXT_LENGTH = 32
SEQUENCE_LENGTH = 20
TRAIN_TOKEN_DATA = "data/TinyStoriesV2-GPT4-train.tokens.bin"
TOKENIZER_NAME = "gpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
  tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
  vocab_size = tokenizer.vocab_size
  dtype = np.uint16 if vocab_size < 65536 else np.uint32

  print(f"===lizhi {vocab_size=} {dtype=} {DEVICE=}")

  transformer_lm = model.TransformerLM(
    vocab_size=vocab_size,
    context_length=CONTEXT_LENGTH,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    d_model=D_MODEL,
    d_ff=None,
    rope_theta=10000,
  )

  optz = optimizer.AdamW(
    params=transformer_lm.parameters(),
    lr=1e-3,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
  )

  big_array = np.memmap(TRAIN_TOKEN_DATA, dtype=dtype, mode="r")

  it = 0
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
    # print(f"{logits=}")
    loss = loss_lib.cross_entropy(logits=logits, targets=y)
    print(f"===lizhi {loss.item()=}")
    loss.backward()
    optz.step()
    it += 1


if __name__ == "__main__":
  main()
