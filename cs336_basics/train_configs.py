import dataclasses
import math

NUM_LAYERS = 2
NUM_HEADS = 4
D_MODEL = 128
BATCH_SIZE = 32
CONTEXT_LENGTH = 32
SEQUENCE_LENGTH = 20


NUM_ITERS = 1000
SAVE_CKPT_EVERY_N_ITERS = 200

EVAL_EVERY_N_ITERS = 20

TOTAL_PRETRAIN_TOKENS = 40960000


@dataclasses.dataclass
class TransformerLMConfig:
  num_layers: int
  num_heads: int
  d_model: int
  d_ff: int | None
  batch_size: int
  context_length: int
  num_iters: int
  rope_theta: int

  lr: float
  weight_decay: float

  save_ckpt_every_n_iters: int
  eval_every_n_iters: int


def _get_num_iters(tokens: int, batch_size: int, context_length: int):
  return math.ceil(tokens / batch_size / context_length)


CONFIG_00001 = TransformerLMConfig(
  num_layers=4,
  num_heads=16,
  d_model=512,
  d_ff=1344,
  batch_size=8,
  context_length=256,
  # Roughly 5K iters for the given TOTAL_PRETRAIN_TOKENS.
  num_iters=_get_num_iters(
    TOTAL_PRETRAIN_TOKENS, batch_size=8, context_length=256
  ),
  rope_theta=10000,
  lr=1e-3,
  weight_decay=0.01,
  save_ckpt_every_n_iters=250,
  eval_every_n_iters=100,
)
