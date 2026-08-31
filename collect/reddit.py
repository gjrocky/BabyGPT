"""
Output
------
  raw/reddit/part_XXXXX.txt

Usage
-----
  python collect/reddit.py
  python collect/reddit.py --limit 200000
  python collect/reddit.py --source eli5 openwebtext
"""
from __future__ import annotations

import argparse
import os

from datasets import load_dataset
from tqdm import tqdm

POSTS_PER_FILE = 1000


def _eli5_parquet_text(x: dict) -> str:
    title = (x.get("title") or "").strip()
    selftext = (x.get("selftext") or "").strip()
    answers = x.get("answers") or {}
    chunks: list[str] = []

    if isinstance(answers, dict):
        texts = answers.get("text")
        if isinstance(texts, list):
            chunks = [str(t).strip() for t in texts[:3] if t and str(t).strip()]
        else:
            a_list = answers.get("a_list")
            if isinstance(a_list, list):
                for a in a_list[:3]:
                    if isinstance(a, dict) and a.get("text"):
                        chunks.append(str(a["text"]).strip())
    if not chunks:
        return ""

    body = "\n\n".join(chunks)
    q = f"Question: {title}"
    if selftext:
        q += f"\n\n{selftext}"
    return f"{q}\n\n{body}"


SOURCES = {
    "eli5": {
        "dataset": "goosmanlei/eli5-parquet",
        "config": None,
        "split": "train",
        "text_fn": _eli5_parquet_text,
        "desc": "ELI5 educational Q&A (Parquet mirror)",
    },
    "openwebtext": {
        "dataset": "Skylion007/openwebtext",
        "config": "plain_text",
        "split": "train",
        "text_fn": lambda x: x["text"],
        "desc": "OpenWebText (high-quality Reddit outbound links)",
    },
    "tifu": {
        "dataset": "Oguzz07/reddit-tifu-dataset",
        "config": None,
        "split": "train",
        "text_fn": lambda x: (
            f"{x.get('instruction', '').strip()}\n\n{x.get('response', '').strip()}"
        ),
        "desc": "Reddit TIFU-style posts (instruction + response; ~1k rows)",
    },
}


def collect_source(name: str, cfg: dict, out_dir: str, limit: int | None) -> None:
    print(f"\nLoading {cfg['desc']}...")
    try:
        if cfg.get("config") is not None:
            ds = load_dataset(
                cfg["dataset"],
                cfg["config"],
                split=cfg["split"],
                streaming=True,
            )
        else:
            ds = load_dataset(
                cfg["dataset"],
                split=cfg["split"],
                streaming=True,
            )
    except Exception as e:
        print(f"  Could not load {name}: {e}")
        return

    src_dir  = os.path.join(out_dir, name)
    os.makedirs(src_dir, exist_ok=True)

    buffer   = []
    file_idx = 0
    total    = 0
    bar      = tqdm(desc=f"  {name}", unit=" posts")

    for item in ds:
        try:
            text = cfg["text_fn"](item).strip()
        except Exception:
            continue

        if len(text) < 200:
            continue

        buffer.append(text)
        total += 1
        bar.update(1)

        if len(buffer) >= POSTS_PER_FILE:
            _flush(buffer, src_dir, file_idx)
            buffer   = []
            file_idx += 1

        if limit and total >= limit:
            break

    if buffer:
        _flush(buffer, src_dir, file_idx)

    bar.close()
    print(f"  Saved {total:,} posts → {src_dir}/")


def _flush(buffer: list[str], out_dir: str, idx: int) -> None:
    path = os.path.join(out_dir, f"part_{idx:05d}.txt")
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        f.write("\n\n---\n\n".join(buffer))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out",    default="raw/reddit",       help="Output directory")
    p.add_argument("--limit",  type=int, default=None,     help="Max posts per source")
    p.add_argument("--source", nargs="*", default=list(SOURCES.keys()),
                   help=f"Sources to download: {', '.join(SOURCES.keys())}")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    for name in args.source:
        if name in SOURCES:
            collect_source(name, SOURCES[name], args.out, args.limit)
        else:
            print(f"Unknown source: {name}. Available: {', '.join(SOURCES.keys())}")

    print(f"\nAll sources done.  Output: {args.out}/")
    print("\nFor full Reddit comments/posts (pre-2023), see:")
    print("  https://academictorrents.com/details/9c263fc85366c1ef8f5bb9da0203f4c8c8db75f4")


if __name__ == "__main__":
    main()
