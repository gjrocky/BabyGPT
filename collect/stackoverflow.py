"""
IMPORTANT: Run this before calling this script:
  pip install py7zr requests tqdm

Usage
-----
  python collect/stackoverflow.py                        # full download + parse
  python collect/stackoverflow.py --xml Posts.xml       # parse already-downloaded XML
  python collect/stackoverflow.py --limit 500000        # first 500K Q&A pairs
"""
from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from io import BytesIO

import requests
from tqdm import tqdm

DUMP_URL          = "https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z"
QA_PER_FILE       = 1000
MIN_ANSWER_SCORE  = 3     # only keep answers with score >= this
MIN_BODY_LEN      = 100   # characters


def download_dump(dest: str) -> str:
    """Stream-download the 7z archive.  Returns path to the downloaded file."""
    path = os.path.join(dest, "stackoverflow-Posts.7z")
    if os.path.exists(path):
        print(f"  Archive already exists at {path}, skipping download.")
        return path

    print(f"Downloading Stack Overflow data dump (~22 GB)...")
    print(f"  URL: {DUMP_URL}")
    r = requests.get(DUMP_URL, stream=True)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    bar   = tqdm(total=total, unit="B", unit_scale=True, desc="Downloading")
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            bar.update(len(chunk))
    bar.close()
    return path


def extract_dump(archive_path: str, dest: str) -> str:
    """Extract the 7z archive.  Returns path to the extracted Posts.xml."""
    xml_path = os.path.join(dest, "Posts.xml")
    if os.path.exists(xml_path):
        print(f"  Posts.xml already extracted at {xml_path}")
        return xml_path

    print("Extracting archive (~85 GB uncompressed, takes a while)...")
    import py7zr
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        z.extractall(path=dest)
    return xml_path


def parse_posts(xml_path: str, out_dir: str, limit: int | None) -> None:
    """
    Parse Posts.xml and write Q&A pairs as plain text.

    The XML has two relevant PostTypeId values:
      1 = Question
      2 = Answer

    Strategy:
      - First pass: collect all accepted-answer pairs (question body + answer body).
      - Write them in a format suitable for both pre-training and instruction tuning.
    """
    print(f"Parsing {xml_path} — this takes 20–60 min for the full file...")

    # First pass: index questions by ID and find their accepted-answer IDs
    questions: dict[str, dict] = {}   # post_id -> {title, body, accepted_answer_id}
    answers:   dict[str, str]  = {}   # post_id -> body

    bar = tqdm(desc="Reading XML", unit=" posts")
    for _, elem in ET.iterparse(xml_path, events=["end"]):
        if elem.tag != "row":
            elem.clear()
            continue

        post_type = elem.get("PostTypeId")
        post_id   = elem.get("Id", "")
        body      = elem.get("Body", "").strip()

        if len(body) < MIN_BODY_LEN:
            elem.clear()
            bar.update(1)
            continue

        if post_type == "1":  # Question
            accepted = elem.get("AcceptedAnswerId")
            if accepted:
                questions[post_id] = {
                    "title":    elem.get("Title", ""),
                    "body":     body,
                    "accepted": accepted,
                }
        elif post_type == "2":  # Answer
            score = int(elem.get("Score", "0"))
            if score >= MIN_ANSWER_SCORE:
                answers[post_id] = body

        elem.clear()
        bar.update(1)
        if limit and len(questions) >= limit * 2:
            break

    bar.close()
    print(f"  Found {len(questions):,} questions with accepted answers, "
          f"{len(answers):,} high-score answers")

    # Second pass: match questions to their accepted answers
    print("Writing Q&A pairs...")
    pairs = []
    for q in questions.values():
        ans = answers.get(q["accepted"])
        if ans:
            pairs.append((q["title"], q["body"], ans))

    if limit:
        pairs = pairs[:limit]

    os.makedirs(out_dir, exist_ok=True)
    file_idx = 0
    buffer   = []

    for title, question, answer in tqdm(pairs, desc="Saving", unit=" pairs"):
        # Strip HTML tags (basic)
        import re
        clean = lambda s: re.sub(r"<[^>]+>", "", s).strip()
        block = (
            f"Question: {clean(title)}\n\n"
            f"{clean(question)}\n\n"
            f"Answer:\n{clean(answer)}"
        )
        buffer.append(block)

        if len(buffer) >= QA_PER_FILE:
            _flush(buffer, out_dir, file_idx)
            buffer   = []
            file_idx += 1

    if buffer:
        _flush(buffer, out_dir, file_idx)

    print(f"\nDone.  {len(pairs):,} Q&A pairs saved to {out_dir}/")


def _flush(buffer: list[str], out_dir: str, idx: int) -> None:
    path = os.path.join(out_dir, f"part_{idx:05d}.txt")
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        f.write("\n\n---\n\n".join(buffer))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out",   default="raw/stackoverflow", help="Output directory")
    p.add_argument("--xml",   default=None,                help="Path to already-downloaded Posts.xml")
    p.add_argument("--limit", type=int, default=None,      help="Max Q&A pairs to extract")
    args = p.parse_args()

    raw_dir = os.path.join(args.out, "_download")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    if args.xml:
        xml_path = args.xml
    else:
        archive  = download_dump(raw_dir)
        xml_path = extract_dump(archive, raw_dir)

    parse_posts(xml_path, args.out, args.limit)


if __name__ == "__main__":
    main()
