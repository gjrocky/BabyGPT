"""
Usage
-----
  python collect/wikipedia.py
  python collect/wikipedia.py --limit 100000    # first 100K articles
  python collect/wikipedia.py --date 20231101   # specific dump date (config {date}.en on Hub)
"""
from __future__ import annotations

import argparse
import os

from datasets import load_dataset
from tqdm import tqdm

ARTICLES_PER_FILE = 500


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out",   default="raw/wikipedia",  help="Output directory")
    p.add_argument(
        "--date",
        default="20231101",
        help="Dump date (YYYYMMDD); Hub config is {date}.en for wikimedia/wikipedia",
    )
    p.add_argument("--limit", type=int, default=None,    help="Max articles to save")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading Wikipedia dump ({args.date}) — this downloads ~3 GB the first time...")
    # streaming=True means articles are fetched one at a time
    ds = load_dataset(
        "wikimedia/wikipedia",
        f"{args.date}.en",
        split="train",
        streaming=True,
    )

    buffer    = []
    file_idx  = 0
    total     = 0
    bar       = tqdm(desc="Saving articles", unit=" articles")

    for article in ds:
        text = article["text"].strip()
        if len(text) < 200:   # skip stubs
            continue

        # Wrap each article with a clear document boundary so the model
        # learns that articles are independent documents, not continuous text.
        buffer.append(f"<|doc|>\n{article['title']}\n\n{text}\n")
        total += 1
        bar.update(1)

        if len(buffer) >= ARTICLES_PER_FILE:
            _flush(buffer, args.out, file_idx)
            buffer   = []
            file_idx += 1

        if args.limit and total >= args.limit:
            break

    if buffer:
        _flush(buffer, args.out, file_idx)

    bar.close()
    print(f"\nDone.  {total:,} articles saved to {args.out}/")


def _flush(buffer: list[str], out_dir: str, idx: int) -> None:
    path = os.path.join(out_dir, f"part_{idx:05d}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(buffer))


if __name__ == "__main__":
    main()
