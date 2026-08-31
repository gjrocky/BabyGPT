"""
Output
------
  data/train.bin    ~90% of tokens
  data/val.bin      ~10% of tokens
  data/tokenizer.json    BPE tokenizer state (load with BPETokenizer.from_state)
  data/stats.json   corpus statistics

Usage
-----
  # After running the collect scripts:
  python collect/prepare.py

  # Custom options:
  python collect/prepare.py --raw raw/ --out data/ --vocab 32000
  python collect/prepare.py --sample-mb 500 --val-ratio 0.05 --workers 16
"""
from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import random

import numpy as np
from tqdm import tqdm

# Add project root to path so we can import data.py
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import BPETokenizer

DTYPE      = np.uint16   # supports vocab_size up to 65535
CHUNK_SIZE = 1 << 20



def find_text_files(raw_dir: str) -> list[str]:
    """Recursively find all .txt files under raw_dir, shuffled."""
    paths = glob.glob(os.path.join(raw_dir, "**", "*.txt"), recursive=True)
    paths = [os.path.abspath(p) for p in paths]   # absolute paths for worker processes
    random.shuffle(paths)
    print(f"Found {len(paths):,} text files in {raw_dir}/")
    return paths



def train_tokenizer(paths: list[str], vocab_size: int, sample_mb: int) -> BPETokenizer:
    """
    Train BPE on a random sample of the corpus.

    We don't need the full corpus to train a good tokenizer — a representative
    100–500 MB sample is enough to find the most frequent byte pairs.
    Sampling also keeps tokenizer training time under an hour.
    """
    target_bytes = sample_mb * 1024 * 1024
    collected    = []
    total        = 0

    print(f"Sampling up to {sample_mb} MB of text for tokenizer training...")
    for path in tqdm(paths, desc="Reading sample"):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            chunk = f.read(CHUNK_SIZE)
            collected.append(chunk)
            total += len(chunk.encode("utf-8"))
        if total >= target_bytes:
            break

    sample = "\n".join(collected)
    print(f"  Sample size: {len(sample.encode('utf-8')) / 1e6:.1f} MB")

    print(f"Training BPE tokenizer (vocab_size={vocab_size})...")
    tok = BPETokenizer()
    tok.train(sample, vocab_size=vocab_size, verbose=True)
    return tok


_worker_tok: BPETokenizer | None = None


def _worker_init(state: dict) -> None:
    global _worker_tok
    _worker_tok = BPETokenizer.from_state(state)


def _encode_file(path: str) -> tuple[str, bytes, bool]:
    """Worker: encode one file and return (path, raw_token_bytes, skipped)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return path, b"", True
    ids = _worker_tok.encode(text)
    return path, np.array(ids, dtype=DTYPE).tobytes(), False


def encode_corpus(
    paths:     list[str],
    tokenizer: BPETokenizer,
    out_dir:   str,
    val_ratio: float,
    n_workers: int,
) -> dict:
    """
    Encode every text file in parallel and write tokens to train.bin / val.bin.

    Uses a multiprocessing pool so all CPU cores encode simultaneously.
    Results are written to disk as they arrive, keeping memory usage low.
    """
    n_val   = max(1, int(len(paths) * val_ratio))
    val_set = set(paths[-n_val:])

    train_path = os.path.join(out_dir, "train.bin")
    val_path   = os.path.join(out_dir, "val.bin")

    train_tokens = val_tokens = skipped = 0

    state = tokenizer.state()
    ctx   = mp.get_context("spawn")

    print(f"Encoding corpus with {n_workers} workers...")
    with open(train_path, "wb") as tf, open(val_path, "wb") as vf:
        with ctx.Pool(n_workers, initializer=_worker_init, initargs=(state,)) as pool:
            for path, buf, was_skipped in tqdm(
                pool.imap_unordered(_encode_file, paths, chunksize=8),
                total=len(paths),
                desc="Encoding corpus",
                unit=" files",
            ):
                if was_skipped:
                    skipped += 1
                    continue
                n_tok = len(buf) // np.dtype(DTYPE).itemsize
                if path in val_set:
                    vf.write(buf)
                    val_tokens += n_tok
                else:
                    tf.write(buf)
                    train_tokens += n_tok

    total_files = len(paths)
    if skipped:
        pct = 100 * skipped / total_files
        print(f"\nSkipped {skipped:,} / {total_files:,} files ({pct:.1f}%) — unreadable on Windows.")
        if pct > 10:
            print("  WARNING: >10% of files skipped. Consider re-collecting the github data on Windows.")
    else:
        print(f"\nAll {total_files:,} files encoded successfully.")

    return {
        "train_tokens": train_tokens,
        "val_tokens":   val_tokens,
        "total_tokens": train_tokens + val_tokens,
        "train_path":   train_path,
        "val_path":     val_path,
        "dtype":        str(DTYPE),
        "vocab_size":   tokenizer.vocab_size,
    }



def main() -> None:
    p = argparse.ArgumentParser(description="Prepare tokenized training data")
    p.add_argument("--raw",       default="raw",        help="Root directory of collected text files")
    p.add_argument("--out",       default="data",       help="Output directory for .bin files")
    p.add_argument("--vocab",     type=int, default=32_000,
                   help="BPE vocabulary size (default 32000, GPT-2 uses 50257)")
    p.add_argument("--sample-mb", type=int, default=200,
                   help="MB of text to sample for tokenizer training")
    p.add_argument("--val-ratio", type=float, default=0.1,
                   help="Fraction of files to use as validation (default 0.1)")
    p.add_argument("--workers",   type=int, default=max(1, mp.cpu_count() - 1),
                   help="Parallel workers for corpus encoding (default: CPU count - 1)")
    p.add_argument("--skip-tokenizer", action="store_true",
                   help="Skip tokenizer training and load existing data/tokenizer.json instead")
    p.add_argument("--seed",      type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)


    paths = find_text_files(args.raw)
    if not paths:
        raise SystemExit(f"No .txt files found under {args.raw}/\nRun the collect scripts first.")

    tok_path = os.path.join(args.out, "tokenizer.json")


    if args.skip_tokenizer:
        if not os.path.exists(tok_path):
            raise SystemExit(f"--skip-tokenizer set but {tok_path} not found. Run without the flag first.")
        print(f"Loading existing tokenizer from {tok_path} ...")
        with open(tok_path) as f:
            tokenizer = BPETokenizer.from_state(json.load(f))
        print(f"  vocab_size={tokenizer.vocab_size}")
    else:
        tokenizer = train_tokenizer(paths, vocab_size=args.vocab, sample_mb=args.sample_mb)
        with open(tok_path, "w") as f:
            json.dump(tokenizer.state(), f)
        print(f"Tokenizer saved → {tok_path}")


    print("\nEncoding full corpus (this is the slow part — get a coffee)...")
    stats = encode_corpus(paths, tokenizer, args.out, args.val_ratio, args.workers)


    stats_path = os.path.join(args.out, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"""
=== Preparation complete ===
  Train tokens : {stats['train_tokens']:>15,}
  Val tokens   : {stats['val_tokens']:>15,}
  Total tokens : {stats['total_tokens']:>15,}
  Vocab size   : {stats['vocab_size']:>15,}
  Train file   : {stats['train_path']}
  Val file     : {stats['val_path']}

Next step:
  python train.py --train-bin {stats['train_path']} \\
                  --val-bin   {stats['val_path']}   \\
                  --tokenizer {tok_path}
""")


if __name__ == "__main__":
    main()
