#!/usr/bin/env python3
"""
train_tokenizer.py

Train a Byte-Level BPE tokenizer for LinuxSLM.

Input:
    data/dataset.txt

Output:
    tokenizer/
        tokenizer.json
        vocab.json
        merges.txt
        tokenizer_config.json
        special_tokens_map.json
"""

from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
import json
import time

# ==========================================================
# Configuration
# ==========================================================

DATASET_FILE = Path("data/dataset.txt")
OUTPUT_DIR = Path("tokenizer")

VOCAB_SIZE = 32768
MIN_FREQUENCY = 2

SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<eod>",
]

# ==========================================================
# Main
# ==========================================================

def main():

    start = time.time()

    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"{DATASET_FILE} not found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training LinuxSLM Tokenizer")
    print("=" * 60)

    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files=[str(DATASET_FILE)],
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
    )

    # Save tokenizer.json
    tokenizer.save(str(OUTPUT_DIR / "tokenizer.json"))

    # Save vocab.json + merges.txt
    tokenizer.save_model(str(OUTPUT_DIR))

    # ------------------------------------------------------
    # tokenizer_config.json
    # ------------------------------------------------------

    tokenizer_config = {
        "model_max_length": 1024,
        "unk_token": "<unk>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
    }

    with open(
        OUTPUT_DIR / "tokenizer_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(tokenizer_config, f, indent=4)

    # ------------------------------------------------------
    # special_tokens_map.json
    # ------------------------------------------------------

    special_tokens_map = {
        "unk_token": "<unk>",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
    }

    with open(
        OUTPUT_DIR / "special_tokens_map.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(special_tokens_map, f, indent=4)

    elapsed = time.time() - start

    print("\nTraining Complete")
    print("-" * 60)
    print(f"Vocabulary Size : {VOCAB_SIZE:,}")
    print(f"Min Frequency   : {MIN_FREQUENCY}")
    print(f"Output          : {OUTPUT_DIR}")
    print(f"Time            : {elapsed:.2f} sec")
    print("-" * 60)

    print("\nGenerated Files:")

    for file in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {file.name}")


if __name__ == "__main__":
    main()