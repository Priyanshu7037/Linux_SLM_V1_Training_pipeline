import csv
import logging
import math
import random
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.amp import GradScaler, autocast

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            self.enabled = False

        def add_scalar(self, *_: Any, **__: Any) -> None:
            return None

        def flush(self) -> None:
            return None

        def close(self) -> None:
            return None

from config import *
from model import GPTConfig, LinuxSLM


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_backend(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "tensorboard").mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(PROJECT_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if not CSV_LOG.exists():
        with CSV_LOG.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "train_loss",
                    "val_loss",
                    "lr",
                    "grad_norm",
                    "tokens_per_sec",
                    "elapsed_sec",
                    "eta_sec",
                    "gpu_memory_gb",
                ]
            )

    return logger


def load_tokens() -> np.memmap:
    if not TRAIN_BIN.exists():
        raise FileNotFoundError(f"Missing dataset: {TRAIN_BIN}")
    return np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")


def split_tokens(tokens: np.memmap, train_fraction: float = 0.9) -> tuple[int, int]:
    split_idx = int(len(tokens) * train_fraction)
    if split_idx <= BLOCK_SIZE or len(tokens) - split_idx <= BLOCK_SIZE:
        raise ValueError("Dataset is too small for the configured block size")
    return split_idx, len(tokens)


def get_batch(
    tokens: np.memmap,
    split: str,
    train_end: int,
    data_end: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    start = 0 if split == "train" else train_end
    end = train_end if split == "train" else data_end
    high = end - BLOCK_SIZE
    ix = np.random.randint(start, high, size=MICRO_BATCH_SIZE)

    x_np = np.stack([tokens[i : i + BLOCK_SIZE].astype(np.int64) for i in ix])
    y_np = np.stack([tokens[i + 1 : i + BLOCK_SIZE + 1].astype(np.int64) for i in ix])

    pin_memory = device.type == "cuda"
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    if pin_memory:
        x = x.pin_memory()
        y = y.pin_memory()

    return (
        x.to(device, non_blocking=pin_memory),
        y.to(device, non_blocking=pin_memory),
    )


def build_model(device: torch.device) -> LinuxSLM:
    model_config = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
        bias=BIAS,
    )
    model = LinuxSLM(model_config).to(device)
    if USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
    return model


def get_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / max(1, WARMUP_STEPS)
    if step >= MAX_STEPS:
        return MIN_LR
    decay_ratio = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    tokens: np.memmap,
    train_end: int,
    data_end: int,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    model.eval()
    losses = []
    for _ in range(EVAL_ITERS):
        x, y = get_batch(tokens, "val", train_end, data_end, device)
        with autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            _, loss = model(x, y)
        if loss is not None:
            losses.append(loss.detach().float())
    model.train()
    return torch.stack(losses).mean().item()


def gpu_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**3)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return str(timedelta(seconds=seconds))


def config_snapshot() -> dict[str, Any]:
    keys = [
        name
        for name in globals()
        if name.isupper()
        and isinstance(globals()[name], (str, int, float, bool, tuple, list, Path))
    ]
    snapshot = {}
    for key in sorted(keys):
        value = globals()[key]
        snapshot[key] = str(value) if isinstance(value, Path) else value
    return snapshot


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [rng.cpu() for rng in torch.cuda.get_rng_state_all()]
    return state


def _as_cpu_byte_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(dtype=torch.uint8).cpu().contiguous()
    if isinstance(value, (list, tuple)):
        return torch.tensor(value, dtype=torch.uint8, device="cpu").contiguous()
    raise TypeError(f"Unsupported RNG state type: {type(value).__name__}")


def _as_cuda_rng_state_list(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [_as_cpu_byte_tensor(value)]
    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            return [_as_cpu_byte_tensor(value)]
        return [_as_cpu_byte_tensor(item) for item in value]
    if isinstance(value, (list, tuple)):
        return [_as_cpu_byte_tensor(item) for item in value]
    raise TypeError(f"Unsupported CUDA RNG state type: {type(value).__name__}")


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_as_cpu_byte_tensor(state["torch"]))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(_as_cuda_rng_state_list(state["cuda"]))


def save_checkpoint(
    path: Path,
    model: LinuxSLM,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    step: int,
    tokens_seen: int,
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": {"step": step},
        "scaler": scaler.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val_loss": best_val_loss,
        "rng_state": rng_state(),
        "config": config_snapshot(),
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: LinuxSLM,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> tuple[int, int, float]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    restore_rng_state(checkpoint["rng_state"])
    return (
        int(checkpoint.get("step", 0)),
        int(checkpoint.get("tokens_seen", 0)),
        float(checkpoint.get("best_val_loss", float("inf"))),
    )


def cleanup_old_checkpoints() -> None:
    step_paths = sorted(
        CHECKPOINT_DIR.glob("step_*.pt"),
        key=lambda path: int(path.stem.split("_")[1]),
    )
    extra = len(step_paths) - KEEP_LAST_CHECKPOINTS
    for path in step_paths[: max(0, extra)]:
        path.unlink(missing_ok=True)


def append_metrics(
    step: int,
    train_loss: float,
    val_loss: Optional[float],
    lr: float,
    grad_norm: float,
    tokens_per_sec: float,
    elapsed: float,
    eta: float,
    gpu_mem: float,
) -> None:
    with CSV_LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                step,
                train_loss,
                "" if val_loss is None else val_loss,
                lr,
                grad_norm,
                tokens_per_sec,
                elapsed,
                eta,
                gpu_mem,
            ]
        )


def main() -> None:
    seed_everything(SEED)
    device = detect_device()
    configure_backend(device)
    logger = setup_logging()
    writer = SummaryWriter(LOG_DIR / "tensorboard")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tokens = load_tokens()
    train_end, data_end = split_tokens(tokens)

    amp_enabled = device.type == "cuda" and USE_FP16
    scaler = GradScaler(device="cuda", enabled=amp_enabled)

    raw_model = build_model(device)
    logger.info("Parameters: %.2fM", raw_model.num_parameters() / 1e6)

    optimizer = raw_model.configure_optimizer(
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        betas=BETAS,
        device_type=device.type,
    )

    start_step = 0
    tokens_seen = 0
    best_val_loss = float("inf")
    latest_path = CHECKPOINT_DIR / "latest.pt"
    if latest_path.exists():
        start_step, tokens_seen, best_val_loss = load_checkpoint(
            latest_path, raw_model, optimizer, scaler, device
        )
        logger.info("Resumed from %s at step %d", latest_path, start_step)

    train_model: torch.nn.Module = raw_model
    if USE_COMPILE and hasattr(torch, "compile"):
        train_model = torch.compile(raw_model)

    train_model.train()
    running_loss = torch.zeros((), device=device)
    last_log_time = time.time()
    start_time = last_log_time
    last_step = start_step
    completed_step = start_step
    last_grad_norm = torch.zeros((), device=device)
    val_loss: Optional[float] = None

    try:
        for step in range(start_step, MAX_STEPS):
            lr = get_lr(step)
            set_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)

            for _ in range(GRADIENT_ACCUMULATION_STEPS):
                x, y = get_batch(tokens, "train", train_end, data_end, device)
                with autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    _, loss = train_model(x, y)
                    if loss is None:
                        raise RuntimeError("Model did not return a training loss")
                    loss = loss / GRADIENT_ACCUMULATION_STEPS

                scaler.scale(loss).backward()
                running_loss += loss.detach().float()

            if GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), GRAD_CLIP)
                last_grad_norm = grad_norm.detach()

            scaler.step(optimizer)
            scaler.update()

            current_step = step + 1
            completed_step = current_step
            tokens_seen += TOKENS_PER_STEP

            do_log = current_step % LOG_INTERVAL == 0
            do_eval = current_step % EVAL_INTERVAL == 0 or current_step == MAX_STEPS
            do_save = current_step % SAVE_INTERVAL == 0 or current_step == MAX_STEPS

            if do_eval:
                val_loss = estimate_loss(
                    train_model, tokens, train_end, data_end, device, amp_enabled
                )
                writer.add_scalar("loss/val", val_loss, current_step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        CHECKPOINT_DIR / "best.pt",
                        raw_model,
                        optimizer,
                        scaler,
                        current_step,
                        tokens_seen,
                        best_val_loss,
                    )

            if do_save:
                save_checkpoint(
                    latest_path,
                    raw_model,
                    optimizer,
                    scaler,
                    current_step,
                    tokens_seen,
                    best_val_loss,
                )
                save_checkpoint(
                    CHECKPOINT_DIR / f"step_{current_step}.pt",
                    raw_model,
                    optimizer,
                    scaler,
                    current_step,
                    tokens_seen,
                    best_val_loss,
                )
                cleanup_old_checkpoints()

            if do_log:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                now = time.time()
                elapsed = now - start_time
                steps_done = current_step - last_step
                interval = now - last_log_time
                tokens_per_sec = (steps_done * TOKENS_PER_STEP) / max(interval, 1e-9)
                remaining_steps = MAX_STEPS - current_step
                eta = remaining_steps * (elapsed / max(1, current_step - start_step))
                train_loss = (running_loss / max(1, steps_done)).item()
                grad_norm_value = last_grad_norm.item()
                gpu_mem = gpu_memory_gb(device)
                val_loss_text = "N/A" if val_loss is None else f"{val_loss:.4f}"

                append_metrics(
                    current_step,
                    train_loss,
                    val_loss,
                    lr,
                    grad_norm_value,
                    tokens_per_sec,
                    elapsed,
                    eta,
                    gpu_mem,
                )
                writer.add_scalar("loss/train", train_loss, current_step)
                writer.add_scalar("lr", lr, current_step)
                writer.add_scalar("grad_norm", grad_norm_value, current_step)
                writer.add_scalar("perf/tokens_per_sec", tokens_per_sec, current_step)
                writer.add_scalar("perf/gpu_memory_gb", gpu_mem, current_step)
                writer.flush()

                message = (
                    f"\nStep {current_step}\n\n"
                    f"Train Loss : {train_loss:.4f}\n"
                    f"Val Loss   : {val_loss_text}\n"
                    f"LR         : {lr:.2e}\n"
                    f"Tokens/sec : {tokens_per_sec / 1000:.0f}k\n"
                    f"GPU Mem    : {gpu_mem:.2f} GB\n"
                    f"ETA        : {format_duration(eta)}"
                )
                logger.info(message)

                running_loss = torch.zeros((), device=device)
                last_log_time = now
                last_step = current_step

        save_checkpoint(
            latest_path,
            raw_model,
            optimizer,
            scaler,
            MAX_STEPS,
            tokens_seen,
            best_val_loss,
        )
        logger.info("Training complete")

    except KeyboardInterrupt:
        logger.info("Interrupted. Saving latest checkpoint...")
        save_checkpoint(
            latest_path,
            raw_model,
            optimizer,
            scaler,
            completed_step,
            tokens_seen,
            best_val_loss,
        )
        logger.info("Checkpoint saved. Exiting cleanly.")
    finally:
        writer.flush()
        writer.close()
        for handler in logger.handlers:
            handler.flush()


if __name__ == "__main__":
    main()
