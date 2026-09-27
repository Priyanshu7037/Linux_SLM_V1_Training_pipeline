#!/usr/bin/env python3
"""
Interactive chatbot inference for LinuxSLM.

Run:
    python chat.py
"""

import argparse
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from tokenizers import Tokenizer

from config import (
    BIAS,
    BLOCK_SIZE,
    CHECKPOINT_DIR,
    DROPOUT,
    N_EMBD,
    N_HEAD,
    N_LAYER,
    SEED,
    TOKENIZER_FILE,
    VOCAB_SIZE,
)
from model import GPTConfig, LinuxSLM

"""
DEFAULT_SYSTEM_PROMPT = (
    "You are LinuxSLM, a helpful assistant for Linux, shell commands, "
    "system administration, scripting, and general technical questions. "
    "Answer clearly and keep replies concise."
)
"""
DEFAULT_SYSTEM_PROMPT = ("")
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with LinuxSLM from a checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default=CHECKPOINT_DIR / "latest.pt",
        type=Path,
        help="Path to model checkpoint. Defaults to checkpoints/latest.pt.",
    )
    parser.add_argument("--max-new-tokens", default=160, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--top-k", default=50, type=int)
    parser.add_argument(
        "--history-turns",
        default=6,
        type=int,
        help="Number of recent user/assistant turns to keep in the prompt.",
    )
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="Instruction placed at the top of every chat prompt.",
    )
    return parser.parse_args()


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


def load_tokenizer(path: Path) -> Tokenizer:
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {path}")
    return Tokenizer.from_file(str(path))


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_model() -> LinuxSLM:
    model_config = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
        bias=BIAS,
    )
    return LinuxSLM(model_config)


def load_model(checkpoint: dict[str, Any], device: torch.device) -> LinuxSLM:
    if "model" not in checkpoint:
        raise KeyError("Checkpoint is missing required key: model")

    model = build_model()
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def decode(tokenizer: Tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def build_chat_prompt(
    system_prompt: str,
    history: list[tuple[str, str]],
    user_message: str,
    history_turns: int,
) -> str:
    kept_history = history[-history_turns:] if history_turns > 0 else []
    lines = [f"System: {system_prompt}", ""]

    for user_text, assistant_text in kept_history:
        lines.append(f"User: {user_text}")
        lines.append(f"Assistant: {assistant_text}")
        lines.append("")

    lines.append(f"User: {user_message}")
    lines.append("Assistant:")
    return "\n".join(lines)


def clean_reply(text: str, prompt: str) -> str:
    if text.startswith(prompt):
        text = text[len(prompt) :]

    stop_markers = [
        "\nUser:",
        "\nSystem:",
        "\nAssistant:",
        "<eos>",
        "<eod>",
        "<|endoftext|>",
    ]
    stop_positions = [text.find(marker) for marker in stop_markers if marker in text]
    if stop_positions:
        text = text[: min(stop_positions)]

    text = text.replace("<eos>", "").replace("<eod>", "").strip()
    if text.startswith("Assistant:"):
        text = text[len("Assistant:") :].strip()
    return text.strip()


@torch.no_grad()
def generate_reply(
    model: LinuxSLM,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
) -> str:
    encoded = tokenizer.encode(prompt)
    if not encoded.ids:
        raise ValueError("Tokenizer produced no tokens for the chat prompt")

    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        do_sample=temperature > 0,
    )
    decoded = decode(tokenizer, output_ids[0].detach().cpu().tolist())
    return clean_reply(decoded, prompt)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = detect_device()
    tokenizer = load_tokenizer(TOKENIZER_FILE)
    checkpoint = load_checkpoint(args.checkpoint)
    model = load_model(checkpoint, device)

    top_k = args.top_k if args.top_k > 0 else None
    step = checkpoint.get("step", "unknown")

    print("LinuxSLM Chat")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Step       : {step}")
    print(f"Device     : {device}")
    print("Type /exit, /quit, or press Ctrl+C to stop.")
    print()

    history: list[tuple[str, str]] = []
    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_message:
            continue
        if user_message.lower() in {"/exit", "/quit"}:
            print("Goodbye.")
            break
        if user_message.lower() == "/clear":
            history.clear()
            print("History cleared.")
            continue

        prompt = build_chat_prompt(
            system_prompt=args.system_prompt,
            history=history,
            user_message=user_message,
            history_turns=args.history_turns,
        )
        reply = generate_reply(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
        )

        if not reply:
            reply = "[No response generated. Try a shorter prompt or lower temperature.]"

        print(f"Bot: {reply}")
        print()
        history.append((user_message, reply))


if __name__ == "__main__":
    main()
