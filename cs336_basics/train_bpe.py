# Run:
#   uv pip install -e . \
#   uv run cs336_basics/train_bpe.py

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from cs336_basics import constants


if __name__ == "__main__":
    # 1. Set up BPE Model
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    
    # 2. Add the Pre-Tokenizer (Splits text and maps raw bytes to visual characters like Ġ)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    
    # 3. FIX: Add the Decoder (Ensures that when you call tokenizer.decode(), 
    # it safely converts 'Ġday' back into ' day')
    tokenizer.decoder = ByteLevelDecoder()

    # 4. Set up the Trainer
    # FIX: We add initial_alphabet=ByteLevel.alphabet() so the trainer knows 
    # about the base 256 byte tokens right from the start.
    trainer = BpeTrainer(
        vocab_size=10_000, 
        special_tokens=["<|endoftext|>", "[UNK]"],
        initial_alphabet=ByteLevel.alphabet()
    )
    
    # 5. Train
    print("Starting BPE training using Hugging Face tokenizers backend...")
    tokenizer.train(["data/TinyStoriesV2-GPT4-train.txt"], trainer)
    print("Training complete!")

    # 6. Save (This saves everything into a single, comprehensive .json file)
    tokenizer.save(constants.TOKENIZER_FILE)
    print(f"Tokenizer saved successfully to {constants.TOKENIZER_FILE}")