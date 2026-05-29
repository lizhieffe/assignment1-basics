# Run:
#   uv pip install -e . \
#   uv run cs336_basics/decode.py

from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import datetime
from typing import IO, Any, BinaryIO

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from jaxtyping import Bool, Float, Int
from torch import Tensor
from transformers import AutoTokenizer
from einops import rearrange

from cs336_basics import checkpoint, constants, model, optimizer, training
from cs336_basics import loss as loss_lib

NUM_LAYERS = 2
NUM_HEADS = 4
D_MODEL = 128
BATCH_SIZE = 32
CONTEXT_LENGTH = 32
SEQUENCE_LENGTH = 20


TEMPERATURE = 0.8
MAX_DECODING_STEPS = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT = "experiments/20260528234107/ckpts/001000/ckpt"


def decode(
  model: nn.Module, tokenizer, inp: str, temperature: float, device: str | None
) -> tuple[str, list[int]]:
  """Decode."""
  outputs = tokenizer(inp, truncation=False, add_special_tokens=False)
  inp_tokens = torch.tensor(
    outputs["input_ids"], dtype=torch.long, device=device
  )  # (s,)
  inp_tokens = rearrange(inp_tokens, "s -> 1 s")
  out_tokens = []

  it = 0
  while True:
    logits = model(inp_tokens)  # (b, s, vocab_size)
    last_logits = logits[:, -1, :]  # (b, vocab_size)
    probs = torch.softmax(last_logits / temperature, dim=-1)  # (b, vocab_size)
    sampled_index = torch.multinomial(probs, num_samples=1)  # (b=1, 1)
    sampled_index = rearrange(sampled_index, "1 d -> d")  # (1,)
    out_token = sampled_index.item()
    out_tokens.append(out_token)
    it += 1
    if out_token == tokenizer.eos_token_id or it == MAX_DECODING_STEPS:
      break
  return tokenizer.decode(out_tokens), out_tokens


def main():
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
    device=DEVICE,
  )

  # TODO: avoid loading optimizer.
  optz = optimizer.AdamW(
    params=transformer_lm.parameters(),
    lr=1e-3,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8,
  )

  checkpoint.load_checkpoint(CKPT, transformer_lm, optz)
  print(f"CKPT {CKPT} is loaded successfully!")

  out_str, out_tokens = decode(
    transformer_lm,
    tokenizer,
    "What is your name?",
    temperature=TEMPERATURE,
    device=DEVICE,
  )
  print(f"Decoding result: {out_str}")
  print(f"Decoding result tokens: {out_tokens}")
  for t in out_tokens:
    print(f"{t} -> {tokenizer.decode(t)}")


if __name__ == "__main__":
  main()
