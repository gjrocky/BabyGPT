"""
Usage
-----
  python collect/gutenberg.py
  python collect/gutenberg.py --limit 1000      # first 1000 books only
  python collect/gutenberg.py --out raw/gutenberg
  python collect/gutenberg.py --workers 16      # parallel downloads (default: 8)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

API_ROOT  = "https://gutendex.com/books"
TEXT_URL  = "https://www.gutenberg.org/files/{id}/{id}-0.txt"
TEXT_URL2 = "https://www.gutenberg.org/files/{id}/{id}.txt"   # fallback
MIRRORS   = [
    "https://www.gutenberg.org",
    "https://gutenberg.pglaf.org",
]
HEADERS     = {"User-Agent": "llm-training-downloader/1.0 (educational)"}
MAX_RETRIES = 8
CATALOG_CACHE = "raw/gutenberg_catalog.json"

RETRY_STATUSES = {429, 500, 502, 503, 504}


def _get_with_retry(url: str, timeout: int = 90) -> requests.Response:
    """GET with exponential backoff.

    Retries on:
      - Timeouts and connection errors
      - 5xx / 429 server errors (gutendex.com returns 503 under load)
    """
    delay = 3.0
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code in RETRY_STATUSES:
                raise requests.HTTPError(response=r)
            r.raise_for_status()
            return r
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = delay * (2 ** attempt)
            code = getattr(getattr(exc, "response", None), "status_code", "")
            label = f"HTTP {code}" if code else exc.__class__.__name__
            tqdm.write(f"  [{label}] retrying in {wait:.0f}s (attempt {attempt+1}/{MAX_RETRIES})...")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def fetch_book_list(limit: int | None, cache_path: str) -> list[dict]:
    """Page through the Gutendex API and collect book metadata.

    Progress is saved to *cache_path* after every page so a crash or server
    error loses at most one page of results (~32 books).  Re-running the
    script resumes from where it left off.
    """
    # Resume from cached progress if available
    books: list[dict] = []
    resume_url: str | None = f"{API_ROOT}/?languages=en&mime_type=text%2Fplain"

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        books      = cache.get("books", [])
        resume_url = cache.get("next_url")  # None means catalog was fully fetched
        tqdm.write(f"  Resuming catalog from {len(books):,} books already fetched.")
        if resume_url is None:
            tqdm.write("  Catalog already complete — skipping fetch.")
            return books[:limit] if limit else books

    bar = tqdm(desc="Fetching catalog", unit=" pages", initial=len(books) // 32)
    url = resume_url
    while url:
        r    = _get_with_retry(url, timeout=90)
        data = r.json()
        books.extend(data["results"])
        bar.update(1)
        bar.set_postfix(total=len(books))

        next_url = data.get("next")

        # Save progress after every page
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"books": books, "next_url": next_url}, f)

        if limit and len(books) >= limit:
            break
        url = next_url
        time.sleep(0.5)   # be polite to the API

    bar.close()
    return books[:limit] if limit else books


def plain_text_url(book: dict) -> str | None:
    """Return the URL for the best plain-text format, or None."""
    for fmt, url in book.get("formats", {}).items():
        if "text/plain" in fmt and url.endswith(".txt") and "zip" not in url:
            return url
    return None


def download_book(book_id: int, out_dir: str) -> bool:
    """Download one book.  Returns True on success."""
    dest = os.path.join(out_dir, f"{book_id}.txt")
    if os.path.exists(dest):
        return True
    
    for template in [TEXT_URL, TEXT_URL2]:
        url = template.format(id=book_id)
        try:
            r = _get_with_retry(url, timeout=60)
            if len(r.text) > 500:
                with open(dest, "w", encoding="utf-8", errors="replace") as f:
                    f.write(r.text)
                return True
        except requests.RequestException:
            pass
        time.sleep(0.2)

    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out",   default="raw/gutenberg", help="Output directory")
    p.add_argument("--limit", type=int, default=None,  help="Max books to download")
    p.add_argument("--cache", default=CATALOG_CACHE,   help="Path to catalog progress cache (JSON)")
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        metavar="N",
        help="Parallel download threads (default: 8). Use 1 for sequential. "
        "Very high values may trigger rate limits on gutenberg.org.",
    )
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Step 1/2 — fetching Gutenberg catalog via Gutendex API...")
    books = fetch_book_list(args.limit, args.cache)
    print(f"  Found {len(books):,} books")

    workers = max(1, args.workers)
    print(f"Step 2/2 — downloading book texts ({workers} worker{'s' if workers != 1 else ''})...")
    ok = fail = 0
    if workers == 1:
        for book in tqdm(books, unit=" books"):
            if download_book(book["id"], args.out):
                ok += 1
            else:
                fail += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(download_book, b["id"], args.out) for b in books]
            with tqdm(total=len(books), unit=" books") as bar:
                for fut in as_completed(futures):
                    try:
                        success = fut.result()
                    except Exception:
                        success = False
                    if success:
                        ok += 1
                    else:
                        fail += 1
                    bar.update(1)

    existing = len(os.listdir(args.out))
    print(f"\nDone.  {ok} downloaded, {fail} failed.  {existing} total files in {args.out}/")


if __name__ == "__main__":
    main()
