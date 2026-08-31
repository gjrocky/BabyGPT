"""
Prerequisites
-------------
  pip install datasets tqdm
  hf auth login          # only if a dataset is gated

Usage
-----
  python collect/sharegpt.py
  python collect/sharegpt.py --limit 50000
  python collect/sharegpt.py --preset vicuna --limit 20000
  python collect/sharegpt.py --out raw/chat_ultra --limit 100000
  python collect/sharegpt.py --dataset HuggingFaceH4/ultrachat_200k --split train_sft --format messages
"""
from __future__ import annotations

import argparse
import os
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

CONVS_PER_FILE = 200

SHAREGPT_FROM: dict[str, str] = {
    "human": "User",
    "user": "User",
    "gpt": "Assistant",
    "chatgpt": "Assistant",
    "assistant": "Assistant",
    "bing": "Assistant",
    "system": "System",
    "function": "Function",
    "observation": "Observation",
}

PRESETS: dict[str, dict[str, Any]] = {
    "ultrachat": {
        "dataset": "HuggingFaceH4/ultrachat_200k",
        "config": None,
        "split": "train_sft",
        "format": "messages",
    },
    "vicuna": {
        "dataset": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "config": None,
        "split": "train",
        "format": "sharegpt",
    },
}


def _role_to_label(role: str) -> str | None:
    r = role.strip().lower()
    if r in ("user", "human"):
        return "User"
    if r in ("assistant", "gpt", "chatgpt"):
        return "Assistant"
    if r == "system":
        return "System"
    return None


def format_messages(row: dict[str, Any]) -> str | None:
    """UltraChat / Zephyr-style messages[]."""
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return None
    lines: list[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role") or ""
        content = (m.get("content") or "").strip()
        if not content:
            continue
        label = _role_to_label(str(role))
        if label is None:
            continue
        lines.append(f"{label}: {content}")
    if len(lines) < 2:
        return None
    return "\n\n".join(lines)


def format_alpaca(row: dict[str, Any]) -> str | None:
    """Alpaca-style instruction/input/output format and Dolly-style instruction/context/response."""
    instruction = (row.get("instruction") or "").strip()
    output      = (row.get("output") or row.get("response") or "").strip()
    if not instruction or not output:
        return None
    context = (row.get("input") or row.get("context") or "").strip()
    user = f"{instruction}\n\n{context}" if context else instruction
    return f"User: {user}\n\nAssistant: {output}"


def format_orca(row: dict[str, Any]) -> str | None:
    """OpenOrca system_prompt/question/response format."""
    question = (row.get("question") or "").strip()
    response = (row.get("response") or "").strip()
    if not question or not response:
        return None
    system = (row.get("system_prompt") or "").strip()
    user = f"{system}\n\n{question}" if system else question
    return f"User: {user}\n\nAssistant: {response}"


def format_truthfulqa(row: dict[str, Any]) -> str | None:
    """TruthfulQA question/best_answer format."""
    question = (row.get("question") or "").strip()
    answer   = (row.get("best_answer") or "").strip()
    if not question or not answer:
        return None
    return f"User: {question}\n\nAssistant: {answer}"


def format_sharegpt(row: dict[str, Any]) -> str | None:
    """Classic ShareGPT conversations[] with from / value."""
    conv = row.get("conversations")
    if conv is None:
        conv = row.get("conversation")
    if not isinstance(conv, list) or not conv:
        return None
    lines: list[str] = []
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        src = (turn.get("from") or turn.get("role") or "").strip().lower()
        text = (turn.get("value") or turn.get("content") or "").strip()
        if not text:
            continue
        label = SHAREGPT_FROM.get(src) or _role_to_label(src)
        if label is None:
            continue
        lines.append(f"{label}: {text}")
    if len(lines) < 2:
        return None
    return "\n\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Export HF chat data to raw/*.txt")
    p.add_argument(
        "--preset",
        choices=list(PRESETS.keys()),
        default="ultrachat",
        help="ultrachat = HuggingFaceH4/ultrachat_200k; vicuna = ShareGPT Vicuna-style",
    )
    p.add_argument("--dataset", default=None, help="Override Hub dataset id")
    p.add_argument("--config", default=None, help="Config / subset name for load_dataset")
    p.add_argument("--split", default=None, help="Split (default from preset)")
    p.add_argument(
        "--format",
        choices=("messages", "sharegpt", "alpaca", "truthfulqa", "orca", "auto"),
        default=None,
        help="Row layout; auto = try messages then sharegpt then alpaca",
    )
    p.add_argument("--out", default="raw/sharegpt", help="Output directory")
    p.add_argument("--limit", type=int, default=None, help="Max conversations to save")
    args = p.parse_args()

    if args.dataset:
        cfg = {
            "dataset": args.dataset,
            "config": args.config,
            "split": args.split or "train",
            "format": args.format or "auto",
        }
    else:
        cfg = dict(PRESETS[args.preset])
        if args.config is not None:
            cfg["config"] = args.config
        if args.split is not None:
            cfg["split"] = args.split
        if args.format is not None:
            cfg["format"] = args.format

    fmt = cfg["format"]
    ds_id = cfg["dataset"]
    split = cfg["split"]
    config = cfg.get("config")

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading {ds_id} (split={split}, format={fmt}) — streaming...")
    if config is not None:
        ds = load_dataset(ds_id, config, split=split, streaming=True)
    else:
        ds = load_dataset(ds_id, split=split, streaming=True)

    def pick_formatter(row: dict[str, Any]) -> str | None:
        if fmt == "messages":
            return format_messages(row)
        if fmt == "sharegpt":
            return format_sharegpt(row)
        if fmt == "alpaca":
            return format_alpaca(row)
        if fmt == "truthfulqa":
            return format_truthfulqa(row)
        if fmt == "orca":
            return format_orca(row)
        return format_messages(row) or format_sharegpt(row) or format_alpaca(row) or format_truthfulqa(row) or format_orca(row)

    buffer: list[str] = []
    file_idx = 0
    total = 0
    bar = tqdm(desc="Rows", unit=" row")

    for row in ds:
        try:
            text = pick_formatter(row)
        except Exception:
            text = None
        bar.update(1)
        if not text or len(text) < 50:
            continue
        buffer.append(text)
        total += 1
        if len(buffer) >= CONVS_PER_FILE:
            _flush(buffer, args.out, file_idx)
            buffer = []
            file_idx += 1
        if args.limit and total >= args.limit:
            break

    if buffer:
        _flush(buffer, args.out, file_idx)
    bar.close()
    print(f"\nDone.  {total:,} conversations → {args.out}/")


def _flush(buffer: list[str], out_dir: str, idx: int) -> None:
    path = os.path.join(out_dir, f"part_{idx:05d}.txt")
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        f.write("\n\n---\n\n".join(buffer))


if __name__ == "__main__":
    main()
