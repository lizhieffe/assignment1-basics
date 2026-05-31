# Run:
#   uv pip install -e . \
#   uv run cs336_basics/decode.py --config=CONFIG_00001

from __future__ import annotations


import argparse
import numpy as np
import torch
import torch.nn as nn

from einops import rearrange

from cs336_basics import (
  bpe_tokenizer,
  checkpoint,
  model,
  optimizer,
  train_configs,
)

NUM_LAYERS = 2
NUM_HEADS = 4
D_MODEL = 128
BATCH_SIZE = 32
CONTEXT_LENGTH = 32
SEQUENCE_LENGTH = 20


TEMPERATURE = 0.8
MAX_DECODING_STEPS = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT = "experiments/20260530233144/ckpts/000250/ckpt"
PROMPT = "What is your name?"
# PROMPT = "你叫什么名字？"

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
    # sampled_index = rearrange(sampled_index, "1 d -> d")  # (1,)
    out_token = sampled_index.item()
    out_tokens.append(out_token)
    inp_tokens = torch.cat([inp_tokens, sampled_index], dim=-1)

    it += 1
    if out_token == tokenizer.eos_token_id or it == MAX_DECODING_STEPS:
      break
  return tokenizer.decode(out_tokens), out_tokens


def main():
  # Set up argument parsing
  parser = argparse.ArgumentParser(
    description="Run Transformer LM training loop with a specified config."
  )
  parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="The variable name of the TransformerLMConfig instance inside train_configs.py",
  )
  args = parser.parse_args()

  # Dynamic lookup using getattr
  try:
    selected_config = getattr(train_configs, args.config)
  except AttributeError:
    raise ValueError(
      f"Config variable '{args.config}' not found in 'cs336_basics/train_configs.py'."
    )

  # Validate type safely
  if not isinstance(selected_config, train_configs.TransformerLMConfig):
    raise TypeError(
      f"'{args.config}' is not an instance of train_configs.TransformerLMConfig."
    )

  tokenizer = bpe_tokenizer.get_tokenizer()
  vocab_size = tokenizer.vocab_size
  dtype = np.uint16 if vocab_size < 65536 else np.uint32

  print(f"{vocab_size=} {dtype=} {DEVICE=}")

  transformer_lm = model.TransformerLM(
    vocab_size=vocab_size,
    context_length=selected_config.context_length,
    num_layers=selected_config.num_layers,
    num_heads=selected_config.num_heads,
    d_model=selected_config.d_model,
    d_ff=selected_config.d_ff,
    rope_theta=selected_config.rope_theta,
    device=DEVICE,
  )

  # TODO: avoid load optimizer.
  optz = optimizer.AdamW(
    params=transformer_lm.parameters(),
    lr=selected_config.lr,
    weight_decay=selected_config.weight_decay,
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
