"""
Prerequisites
-------------
  pip install datasets tqdm

Usage
-----
  python collect/openassistant.py
  python collect/openassistant.py --out raw/oasst --limit 5000
  python collect/openassistant.py --lang all --include-synthetic
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Any

from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

CONVS_PER_FILE = 200


def _rank(m: dict[str, Any]) -> int | float:
    r = m.get("rank")
    return 0 if r is None else r


def _role_label(role: str) -> str | None:
    r = (role or "").strip().lower()
    if r == "prompter":
        return "User"
    if r == "assistant":
        return "Assistant"
    return None


def _ordered_messages(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {m["message_id"]: m for m in msgs}
    ids = set(by_id)

    def is_root(m: dict[str, Any]) -> bool:
        pid = m.get("parent_id")
        if pid is None or pid == "":
            return True
        return pid not in ids

    roots = [m for m in msgs if is_root(m)]
    if not roots:
        return []

    roots.sort(key=lambda x: (_rank(x), str(x.get("message_id", ""))))

    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()

    def dfs(mid: str) -> None:
        if mid in seen or mid not in by_id:
            return
        seen.add(mid)
        ordered.append(by_id[mid])
        children = [c for c in msgs if c.get("parent_id") == mid]
        children.sort(key=lambda c: (_rank(c), str(c.get("message_id", ""))))
        for c in children:
            dfs(c["message_id"])

    for r in roots:
        dfs(r["message_id"])

    leftover = [m for m in msgs if m["message_id"] not in seen]
    if leftover:
        leftover.sort(key=lambda x: (_rank(x), str(x.get("message_id", ""))))
        ordered.extend(leftover)

    return ordered


def _tree_to_text(
    msgs: list[dict[str, Any]],
    lang: str | None,
    include_synthetic: bool,
) -> str | None:
    filtered: list[dict[str, Any]] = []
    for m in msgs:
        if m.get("deleted"):
            continue
        if not include_synthetic and m.get("synthetic"):
            continue
        if lang is not None:
            if (m.get("lang") or "").lower() != lang.lower():
                continue
        filtered.append(m)

    if len(filtered) < 2:
        return None

    ordered = _ordered_messages(filtered)
    lines: list[str] = []
    for m in ordered:
        label = _role_label(str(m.get("role", "")))
        text = (m.get("text") or "").strip()
        if label is None or not text:
            continue
        lines.append(f"{label}: {text}")

    if len(lines) < 2:
        return None
    return "\n\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Export OpenAssistant/oasst1 to raw/*.txt")
    p.add_argument("--out", default="raw/openassistant", help="Output directory")
    p.add_argument("--limit", type=int, default=None, help="Max conversation trees to write")
    p.add_argument(
        "--lang",
        default="en",
        help='Language filter (default: en). Use "all" for every language.',
    )
    p.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Include synthetic (model-generated) messages (default: exclude)",
    )
    args = p.parse_args()

    lang = None if (args.lang or "").lower() == "all" else args.lang

    print("Loading OpenAssistant/oasst1 (train + validation)...")
    ds_dict = load_dataset("OpenAssistant/oasst1")
    combined = concatenate_datasets([ds_dict["train"], ds_dict["validation"]])
    print(f"  {len(combined):,} messages — grouping by message_tree_id...")

    by_tree: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tqdm(combined, desc="Grouping", unit=" msg"):
        tid = row.get("message_tree_id")
        if tid is None:
            continue
        by_tree[str(tid)].append(dict(row))

    print(f"  {len(by_tree):,} trees")

    os.makedirs(args.out, exist_ok=True)

    buffer: list[str] = []
    file_idx = 0
    total = 0

    for _, msgs in tqdm(sorted(by_tree.items()), desc="Encoding trees", unit=" tree"):
        text = _tree_to_text(msgs, lang, args.include_synthetic)
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

    print(f"\nDone.  {total:,} conversations → {args.out}/")


def _flush(buffer: list[str], out_dir: str, idx: int) -> None:
    path = os.path.join(out_dir, f"part_{idx:05d}.txt")
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        f.write("\n\n---\n\n".join(buffer))


if __name__ == "__main__":
    main()
