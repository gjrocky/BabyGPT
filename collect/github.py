"""
Usage
-----
  python collect/github.py --languages python javascript
  python collect/github.py --languages python --limit 500000
"""
from __future__ import annotations

import argparse
import os

from datasets import load_dataset
from tqdm import tqdm

FILES_PER_PART = 500

# Languages available in The Stack — use the exact string from the dataset.
AVAILABLE_LANGUAGES = [
    "python", "javascript", "typescript", "java", "go", "rust",
    "c", "cpp", "c-sharp", "php", "ruby", "scala", "swift", "kotlin",
    "r", "lua", "shell", "sql", "html", "css",
]


def collect_language(lang: str, out_root: str, limit: int | None) -> None:
    out_dir = os.path.join(out_root, lang)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nStreaming {lang} from The Stack...")
    ds = load_dataset(
        "bigcode/the-stack",
        data_dir=f"data/{lang}",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    buffer   = []
    file_idx = 0
    total    = 0
    bar      = tqdm(desc=f"  {lang}", unit=" files")

    for sample in ds:
        content = sample.get("content", "").strip()
        if not content or len(content) < 100:
            continue

        filename = sample.get("path", "unknown")
        buffer.append(f"# {filename}\n{content}\n")
        total += 1
        bar.update(1)

        if len(buffer) >= FILES_PER_PART:
            _flush(buffer, out_dir, file_idx)
            buffer   = []
            file_idx += 1

        if limit and total >= limit:
            break

    if buffer:
        _flush(buffer, out_dir, file_idx)

    bar.close()
    print(f"  Saved {total:,} files → {out_dir}/")


def _flush(buffer: list[str], out_dir: str, idx: int) -> None:
    path = os.path.join(out_dir, f"part_{idx:05d}.txt")
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        f.write("\n\n".join(buffer))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out",       default="raw/github",     help="Output root directory")
    p.add_argument("--languages", nargs="+",
                   default=["python", "javascript"],
                   help=f"Languages to download. Available: {', '.join(AVAILABLE_LANGUAGES)}")
    p.add_argument("--limit",     type=int, default=None,   help="Max files per language")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    invalid = [l for l in args.languages if l not in AVAILABLE_LANGUAGES]
    if invalid:
        print(f"Warning: unknown languages: {invalid}")
        print(f"Available: {', '.join(AVAILABLE_LANGUAGES)}")

    for lang in args.languages:
        if lang in AVAILABLE_LANGUAGES:
            collect_language(lang, args.out, args.limit)

    print(f"\nAll languages done.  Output: {args.out}/")


if __name__ == "__main__":
    main()
