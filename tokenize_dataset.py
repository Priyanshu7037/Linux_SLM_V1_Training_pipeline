#!/usr/bin/env python3
"""
tokenize_dataset.py

Tokenize dataset.txt into train.bin using the trained tokenizer.

Input:
    data/dataset.txt
    tokenizer/tokenizer.json

Output:
    data/train.bin
    data/meta.json
"""

import json
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

# =====================================================
# Configuration
# =====================================================

DATASET_FILE = Path("data/dataset.txt")
TOKENIZER_FILE = Path("tokenizer/tokenizer.json")

OUTPUT_BIN = Path("data/train.bin")
OUTPUT_META = Path("data/meta.json")

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB


# =====================================================
# Main
# =====================================================

def main():

    start = time.time()

    if not DATASET_FILE.exists():
        raise FileNotFoundError(DATASET_FILE)

    if not TOKENIZER_FILE.exists():
        raise FileNotFoundError(TOKENIZER_FILE)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))

    vocab_size = tokenizer.get_vocab_size()

    if vocab_size <= 65535:
        dtype = np.uint16
        dtype_name = "uint16"
    else:
        dtype = np.uint32
        dtype_name = "uint32"

    total_tokens = 0

    print("=" * 60)
    print("Tokenizing Dataset")
    print("=" * 60)

    with DATASET_FILE.open("r", encoding="utf-8") as infile, \
         OUTPUT_BIN.open("wb") as outfile:

        chunk_count = 0

        while True:

            text = infile.read(CHUNK_SIZE)

            if not text:
                break

            # Avoid cutting UTF-8 characters
            if len(text) == CHUNK_SIZE:

                extra = infile.readline()
                text += extra

            ids = tokenizer.encode(text).ids

            np.asarray(ids, dtype=dtype).tofile(outfile)

            total_tokens += len(ids)
            chunk_count += 1

            print(
                f"\rChunks: {chunk_count:4d} | "
                f"Tokens: {total_tokens:,}",
                end=""
            )

    print()

    meta = {
        "vocab_size": vocab_size,
        "dtype": dtype_name,
        "num_tokens": total_tokens,
        "context_length": 1024
    }

    with OUTPUT_META.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    elapsed = time.time() - start

    size_mb = OUTPUT_BIN.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 60)
    print("Tokenization Complete")
    print("=" * 60)
    print(f"Vocabulary Size : {vocab_size:,}")
    print(f"Output Type     : {dtype_name}")
    print(f"Total Tokens    : {total_tokens:,}")
    print(f"Binary Size     : {size_mb:.2f} MB")
    print(f"train.bin       : {OUTPUT_BIN}")
    print(f"meta.json       : {OUTPUT_META}")
    print(f"Time Taken      : {elapsed:.2f} sec")


if __name__ == "__main__":
    main()