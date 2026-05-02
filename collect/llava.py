from __future__ import annotations

import argparse
import json
import os

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def save_stage1(limit: int, out_dir: str) -> None:
    """Flickr30k — image + caption pairs formatted as User/Assistant turns."""
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    print("Loading Flickr30k (streaming)...")
    ds = load_dataset("nlphuji/flickr30k", split="test", streaming=True)

    conversations = []
    bar = tqdm(total=limit, desc="Flickr30k", unit=" images")

    for i, row in enumerate(ds):
        if limit and i >= limit:
            break

        img_filename = f"{i:06d}.jpg"
        img_path     = os.path.join(img_dir, img_filename)

        try:
            img = row["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img.convert("RGB").save(img_path, "JPEG", quality=90)
        except Exception:
            continue

        caption = row["caption"][0] if isinstance(row["caption"], list) else row["caption"]

        conversations.append({
            "image": img_filename,
            "conversations": [
                {"from": "human", "value": "<image>\nDescribe this image."},
                {"from": "gpt",   "value": caption},
            ],
        })
        bar.update(1)

    bar.close()
    _write_json(conversations, out_dir)
    print(f"\nSaved {len(conversations):,} image-caption pairs → {out_dir}/")


def save_stage2(limit: int, out_dir: str) -> None:
    """HuggingFaceM4/the_cauldron — multi-dataset visual instruction following."""
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    print("Loading the_cauldron/vqav2 (streaming)...")
    ds = load_dataset("HuggingFaceM4/the_cauldron", "vqav2", split="train", streaming=True)

    conversations = []
    bar = tqdm(total=limit, desc="the_cauldron", unit=" rows")

    for i, row in enumerate(ds):
        if limit and i >= limit:
            break

        img_filename = f"{i:06d}.jpg"
        img_path     = os.path.join(img_dir, img_filename)

        try:
            imgs = row.get("images") or []
            if not imgs:
                continue
            img = imgs[0]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img.convert("RGB").save(img_path, "JPEG", quality=90)
        except Exception:
            continue

        texts = row.get("texts", [])
        if not texts:
            continue

        convs = []
        for turn in texts:
            user = (turn.get("user") or "").replace("<image>", "").strip()
            asst = (turn.get("assistant") or "").strip()
            if user and asst:
                convs.append({"from": "human", "value": f"<image>\n{user}" if not convs else user})
                convs.append({"from": "gpt",   "value": asst})

        if len(convs) >= 2:
            conversations.append({"image": img_filename, "conversations": convs})
            bar.update(1)

    bar.close()
    _write_json(conversations, out_dir)
    print(f"\nSaved {len(conversations):,} visual instruction pairs → {out_dir}/")


def _write_json(conversations: list, out_dir: str) -> None:
    path = os.path.join(out_dir, "conversations.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(description="Download vision datasets for VisionBabyGPT")
    p.add_argument("--stage",  type=int, choices=[1, 2], default=1)
    p.add_argument("--limit",  type=int, default=10000)
    p.add_argument("--out",    type=str, default="raw/llava")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.stage == 1:
        save_stage1(args.limit, args.out)
    else:
        save_stage2(args.limit, args.out)


if __name__ == "__main__":
    main()
