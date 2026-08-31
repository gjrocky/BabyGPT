"""
Usage
-----
  python collect/arxiv.py                          # full papers (slow, large)
  python collect/arxiv.py --abstracts-only         # abstracts only (~2 GB)
  python collect/arxiv.py --limit 100000           # first 100K papers
  python collect/arxiv.py --categories cs math     # filter by category
"""
from __future__ import annotations

import argparse
import os

from datasets import load_dataset
from tqdm import tqdm

PAPERS_PER_FILE = 200


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out",            default="raw/arxiv",   help="Output directory")
    p.add_argument("--limit",          type=int, default=None, help="Max papers to save")
    p.add_argument("--abstracts-only", action="store_true",    help="Save abstracts only (faster, smaller)")
    p.add_argument("--categories",     nargs="*", default=None,
                   help="Filter by arXiv category prefix e.g. cs math physics")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading ArXiv dataset from RedPajama — first run downloads ~90 GB...")
    print("(Use --abstracts-only for a quick ~2 GB download)\n")

    ds = load_dataset(
        "togethercomputer/RedPajama-Data-1T",
        "arxiv",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    buffer    = []
    file_idx  = 0
    total     = 0
    skipped   = 0
    bar       = tqdm(desc="Saving papers", unit=" papers")

    for paper in ds:
        text     = paper.get("text", "").strip()
        meta     = paper.get("meta", {})
        category = meta.get("primary_category", "")

        # Category filter
        if args.categories:
            if not any(category.startswith(c) for c in args.categories):
                skipped += 1
                continue

        # Abstracts-only mode: cut everything after the intro
        if args.abstracts_only:
            # Papers typically start with title, authors, abstract.
            # We take everything up to ~2000 chars as a reasonable abstract proxy.
            text = text[:2000].strip()

        if len(text) < 300:
            skipped += 1
            continue

        buffer.append(f"<|doc|>\n{text}\n")
        total += 1
        bar.update(1)

        if len(buffer) >= PAPERS_PER_FILE:
            _flush(buffer, args.out, file_idx)
            buffer   = []
            file_idx += 1

        if args.limit and total >= args.limit:
            break

    if buffer:
        _flush(buffer, args.out, file_idx)

    bar.close()
    print(f"\nDone.  {total:,} papers saved, {skipped:,} skipped.  Output: {args.out}/")


def _flush(buffer: list[str], out_dir: str, idx: int) -> None:
    path = os.path.join(out_dir, f"part_{idx:05d}.txt")
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        f.write("\n\n".join(buffer))


if __name__ == "__main__":
    main()
