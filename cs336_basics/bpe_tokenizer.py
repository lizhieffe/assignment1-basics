from transformers import PreTrainedTokenizerFast
import tokenizers as tokenizers_lib

from cs336_basics import constants


def get_tokenizer():
  """Initializes the custom BPE tokenizer into a Hugging Face wrapper."""

  # 1. Load the raw Rust-backed tokenizer from your file
  backend_tokenizer = tokenizers_lib.Tokenizer.from_file(
    constants.TOKENIZER_FILE
  )

  # 2. Wrap it in the Hugging Face high-level API wrapper
  tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=backend_tokenizer,
    unk_token="[UNK]",
    bos_token="<|endoftext|>",  # Configure these to match your training setup
    eos_token="<|endoftext|>",
    pad_token="<|endoftext|>",
  )

  # 3. Handle model_max_length if your training code requires a specific limit
  # tokenizer.model_max_length = 2048

  return tokenizer
