"""
LinuxSLM v1 Configuration
"""

from pathlib import Path
import torch

# ============================================================
# Project
# ============================================================

PROJECT_NAME = "LinuxSLM"

ROOT_DIR = Path(__file__).parent

DATA_DIR = ROOT_DIR / "data"
TOKENIZER_DIR = ROOT_DIR / "tokenizer"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
LOG_DIR = ROOT_DIR / "logs"

TRAIN_BIN = DATA_DIR / "train.bin"
META_FILE = DATA_DIR / "meta.json"
TOKENIZER_FILE = TOKENIZER_DIR / "tokenizer.json"

# ============================================================
# Hardware
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_COMPILE = True
USE_FP16 = True
USE_GRADIENT_CHECKPOINTING = True

NUM_WORKERS = 0

# ============================================================
# Dataset
# ============================================================

VOCAB_SIZE = 32768
BLOCK_SIZE = 1024

# ============================================================
# Model
# ============================================================

N_LAYER = 12
N_HEAD = 12
N_EMBD = 384

DROPOUT = 0.0
BIAS = False

# ============================================================
# Training
# ============================================================

MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 16

MAX_STEPS = 120000

LEARNING_RATE = 3e-4
MIN_LR = 3e-5

WEIGHT_DECAY = 0.1

BETAS = (0.9, 0.95)

GRAD_CLIP = 1.0

WARMUP_STEPS = 2000

# ============================================================
# Evaluation
# ============================================================

EVAL_INTERVAL = 500
EVAL_ITERS = 100

# ============================================================
# Checkpoint
# ============================================================

SAVE_INTERVAL = 1000

KEEP_LAST_CHECKPOINTS = 5

# ============================================================
# Logging
# ============================================================

LOG_INTERVAL = 50

LOG_FILE = LOG_DIR / "train.log"
CSV_LOG = LOG_DIR / "metrics.csv"

# ============================================================
# Randomness
# ============================================================

SEED = 1337

# ============================================================
# Derived
# ============================================================

EFFECTIVE_BATCH_SIZE = (
    MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
)

TOKENS_PER_STEP = (
    EFFECTIVE_BATCH_SIZE * BLOCK_SIZE
)