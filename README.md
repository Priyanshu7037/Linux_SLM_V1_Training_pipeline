# LinuxSLM

A small language model trained on Linux commands, shell scripting, and system administration knowledge. Built from scratch with PyTorch.

## Model Architecture

- **Architecture**: Decoder-only Transformer (GPT-style)
- **Layers**: 12
- **Heads**: 12
- **Embedding dim**: 384
- **Context window**: 1024 tokens
- **Vocabulary**: 32,768 tokens (BPE tokenizer)
- **Parameters**: ~15M (non-embedding)
- **RoPE**: Rotary positional embeddings
- **Norm**: RMSNorm
- **Activation**: SwiGLU

## Quick Start

### Prerequisites

```bash
pip install torch tokenizers numpy einops
```

### Train

```bash
# Train from scratch
python train.py

# Resume from latest checkpoint
python train.py --resume checkpoints/latest.pt
```

Key training configs (in `config.py`):
- `MAX_STEPS = 120,000`
- `MICRO_BATCH_SIZE = 4`, `GRADIENT_ACCUMULATION_STEPS = 16` (effective batch = 64)
- `LEARNING_RATE = 3e-4` with cosine decay to `3e-5`
- `WARMUP_STEPS = 2,000`
- Mixed precision (FP16) + gradient checkpointing + `torch.compile`

### Evaluate During Training

```bash
# Run test prompts against a checkpoint (saves JSON + CSV to results/test_chat/)
python test_chat.py --checkpoint checkpoints/latest.pt

# Verbose output, more samples
python test_chat.py --checkpoint checkpoints/step_5000.pt --samples 5 --verbose --temperature 0.8

# Custom prompts
python test_chat.py --checkpoint checkpoints/latest.pt --prompts-file my_prompts.json
```

### Benchmark Evaluation

```bash
# Run linux_commands.json benchmark (exact/normalized match, pass@k)
python eval.py --checkpoint checkpoints/latest.pt --samples 10
```

### Interactive Chat

```bash
python chat.py --checkpoint checkpoints/latest.pt --temperature 0.7 --top-k 50
```

Commands: `/exit`, `/quit`, `/clear`

## Project Structure

```
Linux_SLM_V1/
├── config.py              # All hyperparameters & paths
├── model.py               # GPTConfig, LinuxSLM (Transformer + LM head)
├── train.py               # Training loop with AMP, checkpointing, logging
├── prepare_dataset.py     # Download/process training data -> train.bin
├── tokenize_dataset.py    # Tokenize text -> token IDs (uint16)
├── train_tokenizer.py     # Train BPE tokenizer (32k vocab)
├── eval.py                # Benchmark evaluation (linux_commands.json)
├── test_chat.py           # Batch evaluation on test prompts
├── chat.py                # Interactive chat
├── test_generate.py       # Quick generation test
├── test_top20.py          # Top-20 token analysis
├── test_overfit.py        # Overfit single batch test
├── test_alignment.py      # Alignment check
├── test.py                # Misc tests
├── checkpoints/           # Model checkpoints (gitignored)
├── data/                  # Tokenized dataset (gitignored)
├── tokenizer/             # tokenizer.json (gitignored)
├── logs/                  # Training logs (gitignored)
├── results/               # Evaluation outputs (gitignored)
├── benchmarks/            # linux_commands.json
└── archive/               # Old model versions (gitignored)
```

## Training Data

The model is trained on a curated dataset of:
- Linux man pages
- Shell commands and one-liners
- Bash/Zsh scripting patterns
- System administration guides
- Docker, Kubernetes, Git CLI references
- Common troubleshooting Q&A

See `prepare_dataset.py` for data sources and processing.

## Tokenizer

BPE tokenizer trained with `tokenizers` library (HuggingFace):
- Vocab size: 32,768
- Special tokens: `<bos>`, `<eos>`, `<pad>`, `<unk>`
- Saved to `tokenizer/tokenizer.json`

## Checkpoints

Saved to `checkpoints/` every 1000 steps:
- `step_{N}.pt` - Full checkpoint (model, optimizer, scheduler, step, losses)
- `latest.pt` - Symlink to most recent
- `best.pt` - Best validation loss

Each checkpoint contains:
```python
{
    "model": state_dict,
    "optimizer": optimizer_state_dict,
    "scheduler": scheduler_state_dict,
    "step": int,
    "train_loss": float,
    "val_loss": float,
    "best_val_loss": float,
    "config": {...}
}
```

## Configuration

All hyperparameters in `config.py`:
```python
# Model
VOCAB_SIZE = 32768
BLOCK_SIZE = 1024
N_LAYER = 12
N_HEAD = 12
N_EMBD = 384
DROPOUT = 0.0
BIAS = False

# Training
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 16
MAX_STEPS = 120000
LEARNING_RATE = 3e-4
MIN_LR = 3e-5
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0
WARMUP_STEPS = 2000
```

## Logging

- Console: every 50 steps (loss, lr, throughput)
- `logs/train.log` - Full training log
- `logs/metrics.csv` - Step, train_loss, val_loss, lr, tokens/sec

## Results

Evaluation outputs saved to `results/`:
- `test_chat/` - Per-checkpoint generations on test prompts (JSON + CSV)
- `evaluation/` - Benchmark results (JSON + CSV)

## Requirements

- Python 3.10+
- PyTorch 2.0+ (for `scaled_dot_product_attention`, `compile`)
- CUDA GPU recommended (works on CPU/MPS but slower)

## License

MIT