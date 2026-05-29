# Preprocessing - Tokenize the text data.
#
# Run:
#   uv pip install -e . \
#   uv run cs336_basics/tokenize_data_main.py

import os
import multiprocessing as mp
from pathlib import Path
import numpy as np
from cs336_basics import constants
from transformers import AutoTokenizer
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = Path("data")
FILE_PATTERNS = ["TinyStoriesV2*train.txt", "TinyStoriesV2*valid.txt"]

# Use 80% of available CPU cores to maximize throughput
NUM_CORES = max(1, int(mp.cpu_count() * 0.8))
# Group lines into chunks to maximize the efficiency of Rust-based tokenizers
BATCH_CHUNK_SIZE = 2000


def get_tokenizer():
  """Initializes the tokenizer and disables internal length warnings."""
  tokenizer = AutoTokenizer.from_pretrained(constants.TOKENIZER_NAME)
  # TODO: this should be unnecessary anymore.
  # Set max length to an astronomical number to stop Hugging Face truncation warnings
  # tokenizer.model_max_length = int(1e30)
  return tokenizer


def tokenize_batch(batch_lines):
  """Worker target function: Tokenizes a batch of text strings on an isolated CPU core."""
  tokenizer = get_tokenizer()
  # truncation=False guarantees zero text or token loss
  outputs = tokenizer(batch_lines, truncation=False, add_special_tokens=False)

  # Append the End-Of-String token ID to the end of each independent sequence
  eos_id = tokenizer.eos_token_id
  return [ids + [eos_id] for ids in outputs["input_ids"]]


def single_file_line_generator(file_path):
  """Streams valid lines from a single file sequentially to keep RAM footprint low."""
  with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
      stripped = line.strip()
      if stripped:
        yield stripped


def chunked_generator(iterable, chunk_size):
  """Groups incoming elements from a generator stream into fixed-size processing blocks."""
  chunk = []
  for item in iterable:
    chunk.append(item)
    if len(chunk) == chunk_size:
      yield chunk
      chunk = []
  if chunk:
    yield chunk


if __name__ == "__main__":
  # 1. Initialize tokenizer baseline to dynamically determine data types
  main_tokenizer = get_tokenizer()
  vocab_size = main_tokenizer.vocab_size
  dtype = np.uint16 if vocab_size < 65536 else np.uint32

  print(f"--- Parallel Tokenization Pipeline ---")
  print(f"Allocated CPU Cores: {NUM_CORES}")
  print(
    f"Detected Vocab Size: {vocab_size} -> Utilizing NumPy type: {dtype.__name__}\n"
  )

  # Gather all matching files up front across all patterns
  all_matching_files = []
  for pattern in FILE_PATTERNS:
    all_matching_files.extend(list(DATA_DIR.glob(pattern)))

  if not all_matching_files:
    print(f"No files found matching patterns: {FILE_PATTERNS}")
    exit()

  # Process each file individually to produce unique corresponding binary maps
  for file_path in all_matching_files:
    # E.g., 'owt_train.txt' -> 'owt_train.tokens.bin'
    bin_output_file = DATA_DIR / f"{file_path.stem}.tokens.bin"

    print(f"==================================================")
    print(f"PROCESSING FILE: {file_path.name}")
    print(f"Output Target:   {bin_output_file.name}")
    print(f"==================================================")

    # ---------------------------------------------------------
    # PASS 0: Count total lines with an active progress bar
    # ---------------------------------------------------------
    total_lines = 0
    # Since we don't know total lines in advance, tqdm works as an active counter
    with tqdm(desc="Counting total lines", unit=" lines") as line_pbar:
      for _ in single_file_line_generator(file_path):
        total_lines += 1
        line_pbar.update(1)

    total_chunks = (total_lines + BATCH_CHUNK_SIZE - 1) // BATCH_CHUNK_SIZE
    print(f"Total lines: {total_lines:,} ({total_chunks:,} chunks determined)")

    if total_lines == 0:
      print(f"Skipping file because it contains no valid text.\n")
      continue

    # ---------------------------------------------------------
    # PASS 1: Calculate Total Token Count for Memmap Sizing
    # ---------------------------------------------------------
    total_tokens = 0
    with mp.Pool(processes=NUM_CORES) as pool:
      lines_gen = single_file_line_generator(file_path)
      chunks_gen = chunked_generator(lines_gen, BATCH_CHUNK_SIZE)

      progress_bar = tqdm(
        pool.imap(tokenize_batch, chunks_gen),
        total=total_chunks,
        desc="Calculating tokens",
      )
      for tokenized_batch in progress_bar:
        total_tokens += sum(len(tokens) for tokens in tokenized_batch)

    print(f"Total tokens calculated: {total_tokens:,}")

    # ---------------------------------------------------------
    # PASS 2: Allocate Memory-Map File and Stream Binary Tokens
    # ---------------------------------------------------------
    print(f"Allocating continuous space on disk...")
    mmap_array = np.memmap(
      bin_output_file, dtype=dtype, mode="w+", shape=(total_tokens,)
    )

    current_idx = 0
    with mp.Pool(processes=NUM_CORES) as pool:
      lines_gen = single_file_line_generator(file_path)
      chunks_gen = chunked_generator(lines_gen, BATCH_CHUNK_SIZE)

      progress_bar = tqdm(
        pool.imap(tokenize_batch, chunks_gen),
        total=total_chunks,
        desc="Writing binary data",
      )
      for tokenized_batch in progress_bar:
        # Fast matrix flattening via numpy concatenation vectorization
        flat_batch = np.concatenate(
          [np.array(t, dtype=dtype) for t in tokenized_batch]
        )
        batch_len = len(flat_batch)

        # Directly map memory allocation slice to disk sector bounds
        mmap_array[current_idx : current_idx + batch_len] = flat_batch
        current_idx += batch_len

    # Force operating system file system buffers to flush to solid-state memory
    mmap_array.flush()
    del mmap_array
    print(f"Successfully processed! Created: {bin_output_file.name}\n")
