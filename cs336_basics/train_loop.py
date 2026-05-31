# Run:
#   1. uv pip install -e .
#   2. Download training data: https://github.com/stanford-cs336/assignment1-basics#download-data
#   3. tokenize the data: See `cs336_basics/tokenize_data_main.py`
#   4. Run training: `uv run cs336_basics/train_loop.py --config=CONFIG_00001`
#
# Training metrics: https://wandb.ai/lizhieffe-teestory/cs336-assignment-1

from __future__ import annotations

import os
from datetime import datetime

import argparse
import dataclasses
import jaxtyping
import numpy as np
import torch
import torch.nn as nn
import wandb


from cs336_basics import (
  bpe_tokenizer,
  checkpoint,
  model,
  optimizer,
  train_configs,
  training,
)
from cs336_basics import loss as loss_lib


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAINING_QUERYSET = "data/TinyStoriesV2-GPT4-train.tokens.bin"
EVAL_QUERYSET = "data/TinyStoriesV2-GPT4-valid.tokens.bin"
EVAL_QUERY_COUNT = 1024


def run_eval(
  x: jaxtyping.Int[torch.Tensor, "b s"],
  targets: jaxtyping.Int[torch.Tensor, "b s"],
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


def run_training_loop(transformer_lm_config: train_configs.TransformerLMConfig):
  uuid = datetime.now().strftime("%Y%m%d%H%M%S")

  tokenizer = bpe_tokenizer.get_tokenizer()
  vocab_size = tokenizer.vocab_size
  print(f"Tokenizer vocab size: {vocab_size}")
  dtype = np.uint16 if vocab_size < 65536 else np.uint32

  config_dict = dataclasses.asdict(transformer_lm_config)
  print(f"Start training with config: {config_dict}")
  wandb.init(
    project="cs336-assignment-1", name=f"run-{uuid}", config=config_dict
  )

  transformer_lm = model.TransformerLM(
    vocab_size=vocab_size,
    context_length=transformer_lm_config.context_length,
    num_layers=transformer_lm_config.num_layers,
    num_heads=transformer_lm_config.num_heads,
    d_model=transformer_lm_config.d_model,
    d_ff=transformer_lm_config.d_ff,
    rope_theta=transformer_lm_config.rope_theta,
    device=DEVICE,
  )

  # # Track gradients and model topology.
  # wandb.watch(
  #   transformer_lm,
  #   log="gradients",
  #   log_freq=transformer_lm_config.eval_every_n_iters,
  # )

  optz = optimizer.AdamW(
    params=transformer_lm.parameters(),
    lr=transformer_lm_config.lr,
    weight_decay=transformer_lm_config.weight_decay,
    betas=(0.9, 0.999),
    eps=1e-8,
  )

  big_array = np.memmap(TRAINING_QUERYSET, dtype=dtype, mode="r")
  eval_query_array = np.memmap(EVAL_QUERYSET, dtype=dtype, mode="r")

  out_dir = f"experiments/{uuid}/ckpts"
  os.makedirs(out_dir, exist_ok=False)
  print(f"Verified checkpoint directory structure exists at: {out_dir}")

  if (
    transformer_lm_config.save_ckpt_every_n_iters
    and transformer_lm_config.save_ckpt_every_n_iters > 0
  ):
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
    context_length=transformer_lm_config.context_length,
    device=DEVICE,
  )

  eval_loss = run_eval(x=eval_x, targets=eval_targets, model=transformer_lm)
  print(f"Initial CKPT Eval Loss = {eval_loss:.3f}")
  wandb.log({"eval/loss": eval_loss}, step=0)

  it = 1
  for _ in range(transformer_lm_config.num_iters):
    optz.zero_grad()
    x, y = training.sample_training_data(
      x=big_array,
      batch_size=transformer_lm_config.batch_size,
      context_length=transformer_lm_config.context_length,
      device=DEVICE,
    )
    logits = transformer_lm(x)
    loss = loss_lib.cross_entropy(logits=logits, targets=y)
    if it % transformer_lm_config.eval_every_n_iters == 0:
      print(f"============== it = {it} ====================")
      loss_val = loss.item()
      print(f"Train Loss = {loss_val:.3f}")
      wandb.log({"train/loss": loss_val}, step=it)
    loss.backward()
    optz.step()

    if (
      transformer_lm_config.save_ckpt_every_n_iters
      and transformer_lm_config.save_ckpt_every_n_iters > 0
      and it % transformer_lm_config.save_ckpt_every_n_iters == 0
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

    if it % transformer_lm_config.eval_every_n_iters == 0:
      eval_loss = run_eval(x=eval_x, targets=eval_targets, model=transformer_lm)
      print(f"Eval Loss = {eval_loss:.3f}")
      wandb.log({"eval/loss": eval_loss}, step=it)

    it += 1

  wandb.finish()


if __name__ == "__main__":
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

  run_training_loop(transformer_lm_config=selected_config)
