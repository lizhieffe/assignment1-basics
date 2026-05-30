import dataclasses

NUM_LAYERS = 2
NUM_HEADS = 4
D_MODEL = 128
BATCH_SIZE = 32
CONTEXT_LENGTH = 32
SEQUENCE_LENGTH = 20


NUM_ITERS = 1000
SAVE_CKPT_EVERY_N_ITERS = 200

EVAL_EVERY_N_ITERS = 20


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


CONFIG_00001 = TransformerLMConfig(
  num_layers=NUM_LAYERS,
  num_heads=NUM_HEADS,
  d_model=D_MODEL,
  d_ff=None,
  batch_size=BATCH_SIZE,
  context_length=CONTEXT_LENGTH,
  num_iters=NUM_ITERS,
  rope_theta=10000,
  lr=1e-3,
  weight_decay=0.01,
  save_ckpt_every_n_iters=SAVE_CKPT_EVERY_N_ITERS,
  eval_every_n_iters=EVAL_EVERY_N_ITERS,
)
