"""
Output
------
  raw/common_crawl/part_XXXXX.txt

Usage
-----
  python collect/common_crawl.py --limit 1000000     # 1M documents (~5 GB)
  python collect/common_crawl.py --limit 5000000     # 5M documents (~25 GB)
  python collect/common_crawl.py --bytes 10000000000 # stop after 10 GB
"""
from __future__ import annotations

import argparse
import os

from datasets import load_dataset
from tqdm import tqdm

DOCS_PER_FILE   = 2000
MIN_DOC_LEN     = 500
MAX_DOC_LEN     = 50_000


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out",   default="raw/common_crawl", help="Output directory")
    p.add_argument("--limit", type=int,  default=1_000_000,
                   help="Max documents to save")
    p.add_argument("--bytes", type=int,  default=None,
                   help="Stop after accumulating this many bytes of text")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Streaming C4 (en) — no full download needed, processes on the fly...")
    print(f"Target: {args.limit:,} documents\n")

    ds = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    buffer     = []
    file_idx   = 0
    total      = 0
    total_bytes = 0
    bar        = tqdm(desc="Saving documents", unit=" docs")

    for doc in ds:
        text = doc.get("text", "").strip()
        if len(text) < MIN_DOC_LEN:
            continue

        text = text[:MAX_DOC_LEN]   # cap very long pages
        buffer.append(text)
        total       += 1
        total_bytes += len(text.encode("utf-8"))
        bar.update(1)
        bar.set_postfix(gb=f"{total_bytes/1e9:.2f}")

        if len(buffer) >= DOCS_PER_FILE:
            _flush(buffer, args.out, file_idx)
            buffer   = []
            file_idx += 1

        if args.bytes and total_bytes >= args.bytes:
            break
        if total >= args.limit:
            break

    if buffer:
        _flush(buffer, args.out, file_idx)

    bar.close()
    gb = total_bytes / 1e9
    print(f"\nDone.  {total:,} documents  ({gb:.2f} GB)  saved to {args.out}/")


def _flush(buffer: list[str], out_dir: str, idx: int) -> None:
    path = os.path.join(out_dir, f"part_{idx:05d}.txt")
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        f.write("\n\n".join(buffer))


if __name__ == "__main__":
    main()
