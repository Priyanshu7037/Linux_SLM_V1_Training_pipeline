#!/usr/bin/env python3
import argparse
import csv
import json
import random
import re
import shlex
from datetime import datetime
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
    ROOT_DIR,
    SEED,
    TOKENIZER_FILE,
    VOCAB_SIZE,
)
from archive.model import GPTConfig, LinuxSLM


DEFAULT_BENCHMARK = ROOT_DIR / "benchmarks" / "linux_commands.json"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results" / "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LinuxSLM command generation.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, type=Path)
    parser.add_argument("--samples", default=5, type=int)
    parser.add_argument("--max-new-tokens", default=64, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top-k", default=50, type=int)
    parser.add_argument("--seed", default=SEED, type=int)
    parser.add_argument("--incorrect-limit", default=25, type=int)
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


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError("Benchmark must be a JSON list")
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"Benchmark entry {idx} is not an object")
        if not isinstance(item.get("prompt"), str):
            raise TypeError(f"Benchmark entry {idx} is missing prompt")
        if not isinstance(item.get("expected"), list) or not item["expected"]:
            raise TypeError(f"Benchmark entry {idx} is missing expected commands")
    return data


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


def clean_candidate(text: str, prompt: str) -> str:
    text = text.replace("<eod>", "").replace("<eos>", "").strip()
    if text.startswith(prompt):
        text = text[len(prompt) :].strip()
    text = re.sub(r"^```(?:bash|sh|shell)?", "", text.strip(), flags=re.IGNORECASE)
    text = text.replace("```", "").strip()

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("$ "):
            line = line[2:].strip()
        elif line == "$":
            continue
        return line.rstrip(".")
    return text.rstrip(".")


def normalize_command(command: str) -> str:
    command = clean_candidate(command, "")
    command = command.replace("`", "").strip()
    command = re.sub(r"\s+", " ", command)
    command = re.sub(r"\s*([|<>])\s*", r" \1 ", command)
    command = re.sub(r"\s+", " ", command).strip()
    try:
        return " ".join(shlex.split(command, posix=True))
    except ValueError:
        return command


def exact_match(candidate: str, expected: list[str]) -> bool:
    return candidate.strip() in [item.strip() for item in expected]


def normalized_match(candidate: str, expected: list[str]) -> bool:
    candidate_norm = normalize_command(candidate)
    return candidate_norm in {normalize_command(item) for item in expected}


def decode(tokenizer: Tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def generate_one(
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
        raise ValueError(f"Tokenizer produced no tokens for prompt: {prompt!r}")
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        do_sample=temperature > 0,
    )
    decoded = decode(tokenizer, output_ids[0].detach().cpu().tolist())
    return clean_candidate(decoded, prompt)


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def print_results(summary: dict[str, Any], incorrect: list[dict[str, Any]], limit: int) -> None:
    print("=" * 72)
    print("LinuxSLM Benchmark Evaluation")
    print("=" * 72)
    print(f"Checkpoint       : {summary['checkpoint']}")
    print(f"Checkpoint step  : {summary['checkpoint_step']}")
    print(f"Benchmark        : {summary['benchmark']}")
    print(f"Prompts          : {summary['num_prompts']}")
    print(f"Samples/prompt   : {summary['samples_per_prompt']}")
    print(f"Exact Match      : {percent(summary['exact_match'])}")
    print(f"Normalized Match : {percent(summary['normalized_match'])}")
    print(f"Pass@{summary['samples_per_prompt']:<10}: {percent(summary['pass_at_k'])}")
    print()
    print("Category Results")
    print("-" * 72)
    for category, stats in sorted(summary["categories"].items()):
        print(
            f"{category:18} "
            f"n={stats['count']:3d}  "
            f"exact={percent(stats['exact_match']):>7}  "
            f"norm={percent(stats['normalized_match']):>7}  "
            f"pass@k={percent(stats['pass_at_k']):>7}"
        )

    if not incorrect:
        print()
        print("No incorrect examples.")
        return

    print()
    print(f"Incorrect Examples (showing {min(limit, len(incorrect))} of {len(incorrect)})")
    print("-" * 72)
    for item in incorrect[:limit]:
        print(f"Prompt   : {item['prompt']}")
        print(f"Expected : {item['expected']}")
        print("Generated:")
        for idx, output in enumerate(item["generated"], start=1):
            print(f"  {idx}. {output}")
        print("-" * 72)


def save_results(results_dir: Path, payload: dict[str, Any]) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = Path(payload["summary"]["checkpoint"]).stem
    json_path = results_dir / f"{checkpoint_name}_{timestamp}.json"
    csv_path = results_dir / f"{checkpoint_name}_{timestamp}.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["category", "prompt", "expected", "top_sample", "exact", "normalized", "pass_at_k"]
        )
        for item in payload["items"]:
            writer.writerow(
                [
                    item["category"],
                    item["prompt"],
                    " || ".join(item["expected"]),
                    item["generated"][0] if item["generated"] else "",
                    item["exact"],
                    item["normalized"],
                    item["pass_at_k"],
                ]
            )
    return json_path


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = detect_device()
    tokenizer = load_tokenizer(TOKENIZER_FILE)
    benchmark = load_benchmark(args.benchmark)
    checkpoint = load_checkpoint(args.checkpoint)
    model = load_model(checkpoint, device)

    items = []
    category_counts: dict[str, dict[str, float]] = {}

    for index, entry in enumerate(benchmark, start=1):
        prompt = entry["prompt"]
        expected = entry["expected"]
        category = entry.get("category", "uncategorized")

        generated = [
            generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            for _ in range(args.samples)
        ]

        exact = exact_match(generated[0], expected)
        normalized = normalized_match(generated[0], expected)
        pass_at_k = any(normalized_match(sample, expected) for sample in generated)

        stats = category_counts.setdefault(
            category, {"count": 0.0, "exact": 0.0, "normalized": 0.0, "pass_at_k": 0.0}
        )
        stats["count"] += 1
        stats["exact"] += float(exact)
        stats["normalized"] += float(normalized)
        stats["pass_at_k"] += float(pass_at_k)

        items.append(
            {
                "index": index,
                "category": category,
                "prompt": prompt,
                "expected": expected,
                "generated": generated,
                "exact": exact,
                "normalized": normalized,
                "pass_at_k": pass_at_k,
            }
        )

    total = len(items)
    categories = {}
    for category, stats in category_counts.items():
        count = int(stats["count"])
        categories[category] = {
            "count": count,
            "exact_match": stats["exact"] / count,
            "normalized_match": stats["normalized"] / count,
            "pass_at_k": stats["pass_at_k"] / count,
        }

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint.get("step", "unknown"),
        "benchmark": str(args.benchmark),
        "num_prompts": total,
        "samples_per_prompt": args.samples,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "exact_match": sum(float(item["exact"]) for item in items) / total,
        "normalized_match": sum(float(item["normalized"]) for item in items) / total,
        "pass_at_k": sum(float(item["pass_at_k"]) for item in items) / total,
        "categories": categories,
    }
    incorrect = [item for item in items if not item["pass_at_k"]]
    payload = {"summary": summary, "items": items, "incorrect": incorrect}
    result_path = save_results(DEFAULT_RESULTS_DIR, payload)

    print_results(summary, incorrect, args.incorrect_limit)
    print()
    print(f"Saved results: {result_path}")


if __name__ == "__main__":
    main()
