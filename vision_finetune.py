"""
  # Stage 1 — train projection only
  python vision_finetune.py \\
    --checkpoint sft_checkpoint.pt \\
    --data raw/llava \\
    --stage 1 --steps 5000 --lr 1e-3 \\
    --bf16 --out vision_checkpoint.pt

  # Stage 2 — full fine-tune
  python vision_finetune.py \\
    --checkpoint vision_checkpoint.pt \\
    --data raw/llava \\
    --stage 2 --steps 10000 --lr 1e-4 \\
    --bf16 --grad-checkpoint --out vision_checkpoint.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPVisionModel

from data import BPETokenizer
from model import BabyGPT, VisionBabyGPT

CLIP_MODEL   = "openai/clip-vit-base-patch32"
NUM_IMG_TOKS = VisionBabyGPT.NUM_IMG_TOKENS


def load_clip(device: str):
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    encoder   = CLIPVisionModel.from_pretrained(CLIP_MODEL).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return processor, encoder


@torch.no_grad()
def encode_images(encoder, processor, image_paths: list[str], device: str) -> torch.Tensor:
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(images=images, return_tensors="pt").to(device)
    out    = encoder(**inputs)
    feats  = out.last_hidden_state[:, 1:, :]  # remove CLS token → (B, 49, 768)
    return feats


def load_data(data_dir: str) -> list[dict]:
    conv_path = os.path.join(data_dir, "conversations.json")
    with open(conv_path) as f:
        data = json.load(f)
    img_dir = os.path.join(data_dir, "images")
    valid   = []
    for item in data:
        img_path = os.path.join(img_dir, item["image"])
        if os.path.exists(img_path):
            item["image_path"] = img_path
            valid.append(item)
    print(f"  Loaded {len(valid):,} valid image-conversation pairs")
    return valid


def tokenize_conversation(tokenizer, turns, block_size):
    ids:  list[int] = []
    mask: list[int] = []
    for role, content in turns:
        prefix     = f"{role}: "
        suffix     = "\n\n"
        prefix_ids = tokenizer.encode(prefix)
        content_ids = tokenizer.encode(content)
        suffix_ids = tokenizer.encode(suffix)
        turn_ids   = prefix_ids + content_ids + suffix_ids
        ids.extend(turn_ids)
        mask.extend([1 if role == "Assistant" else 0] * len(turn_ids))

    max_len = block_size - NUM_IMG_TOKS - 1
    if len(ids) > max_len:
        ids  = ids[:max_len]
        mask = mask[:max_len]
    return ids, mask


def get_batch(data, tokenizer, encoder, processor, block_size, batch_size, device):
    items = random.choices(data, k=batch_size)

    image_paths = [item["image_path"] for item in items]
    image_feats = encode_images(encoder, processor, image_paths, device)

    max_len = block_size - NUM_IMG_TOKS - 1
    xs, ys, ms = [], [], []

    for item in items:
        turns = [(t["from"].capitalize(), t["value"].replace("<image>", "").strip())
                 for t in item["conversations"]
                 if t["from"] in ("human", "gpt", "user", "assistant")]
        turns = [(("User" if r in ("Human", "User") else "Assistant"), c) for r, c in turns]

        ids, mask = tokenize_conversation(tokenizer, turns, block_size)
        if len(ids) < 2:
            ids  = tokenizer.encode("User: Hello\n\nAssistant: Hi!")
            mask = [0] * (len(ids) // 2) + [1] * (len(ids) - len(ids) // 2)

        pad = max_len - len(ids)
        if pad > 0:
            ids  = ids  + [0] * pad
            mask = mask + [0] * pad

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:],  dtype=torch.long)
        m = torch.tensor(mask[1:], dtype=torch.float)

        xs.append(x)
        ys.append(y)
        ms.append(m)

    return (
        torch.stack(xs).to(device),
        torch.stack(ys).to(device),
        torch.stack(ms).to(device),
        image_feats.to(device),
    )


def get_lr(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    if step >= total:
        return min_lr
    progress = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",    required=True)
    p.add_argument("--data",          required=True)
    p.add_argument("--out",           default="vision_checkpoint.pt")
    p.add_argument("--stage",         type=int, default=1, choices=[1, 2])
    p.add_argument("--steps",         type=int, default=5000)
    p.add_argument("--batch-size",    type=int, default=4)
    p.add_argument("--grad-accum",    type=int, default=4)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--min-lr",        type=float, default=1e-4)
    p.add_argument("--warmup-steps",  type=int, default=200)
    p.add_argument("--grad-clip",     type=float, default=1.0)
    p.add_argument("--eval-every",    type=int, default=500)
    p.add_argument("--save-every",    type=int, default=1000)
    p.add_argument("--bf16",          action="store_true")
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading base checkpoint: {args.checkpoint}")
    ckpt       = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta       = ckpt["meta"]
    tokenizer  = BPETokenizer.from_state(meta["tokenizer"])
    block_size = meta["block_size"]

    base_model = BabyGPT(
        vocab_size             = meta["vocab_size"],
        block_size             = block_size,
        n_layer                = meta["n_layer"],
        n_head                 = meta["n_head"],
        n_embd                 = meta["n_embd"],
        dropout                = meta.get("dropout", 0.0),
        gradient_checkpointing = args.grad_checkpoint,
    )
    base_model.load_state_dict(ckpt["model"])

    # If resuming a vision checkpoint, load existing vision_proj weights
    model = VisionBabyGPT(base_model)
    if "vision_proj" in ckpt:
        model.vision_proj.load_state_dict(ckpt["vision_proj"])
        print("Loaded existing vision_proj weights")

    if args.bf16 and torch.cuda.is_bf16_supported():
        model = model.to(torch.bfloat16)

    model = model.to(args.device)

    # Stage 1: only train projection layer
    # Stage 2: train projection + full LM
    if args.stage == 1:
        for param in model.lm.parameters():
            param.requires_grad = False
        trainable = list(model.vision_proj.parameters())
        print("Stage 1: training projection layer only")
    else:
        for param in model.parameters():
            param.requires_grad = True
        trainable = list(model.parameters())
        print("Stage 2: training projection + language model")

    print(f"  Trainable params: {sum(p.numel() for p in trainable if p.requires_grad):,}")

    print(f"\nLoading CLIP: {CLIP_MODEL}")
    processor, encoder = load_clip(args.device)

    print(f"\nLoading data from: {args.data}")
    data = load_data(args.data)

    opt = torch.optim.AdamW(
        [p for p in trainable if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95),
        fused=True if args.device == "cuda" else False,
    )

    dtype  = torch.bfloat16 if args.bf16 and torch.cuda.is_bf16_supported() else torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    ctx    = (torch.amp.autocast(device_type="cuda", dtype=dtype)
              if args.device == "cuda" else torch.nullcontext())

    print(f"\nVision fine-tuning  stage={args.stage}  steps={args.steps:,}  device={args.device}\n")

    bar        = tqdm(range(args.steps), desc="Vision SFT", unit=" steps")
    accum_loss = 0.0
    best_loss  = float("inf")
    opt.zero_grad(set_to_none=True)

    for step in bar:
        t0 = time.time()

        for _ in range(args.grad_accum):
            x, y, mask, img_feats = get_batch(
                data, tokenizer, encoder, processor,
                block_size, args.batch_size, args.device,
            )
            with ctx:
                logits, _ = model(x, image_features=img_feats)
                text_logits = logits[:, NUM_IMG_TOKS:, :]
                y_masked    = y.masked_fill(mask == 0, -100)
                loss = F.cross_entropy(
                    text_logits.view(-1, text_logits.size(-1)),
                    y_masked.view(-1),
                    ignore_index=-100,
                ) / args.grad_accum
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        lr = get_lr(step, args.warmup_steps, args.steps, args.lr, args.min_lr)
        for pg in opt.param_groups:
            pg["lr"] = lr

        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        dt   = time.time() - t0
        toks = args.batch_size * (block_size - NUM_IMG_TOKS) * args.grad_accum
        bar.set_postfix(loss=f"{accum_loss:.4f}", lr=f"{lr:.2e}", tok_s=f"{toks/dt:.0f}")

        if accum_loss < best_loss:
            best_loss = accum_loss
            _save(model, tokenizer, meta, args, step, best=True)

        accum_loss = 0.0

        if step % args.save_every == 0 and step > 0:
            _save(model, tokenizer, meta, args, step)

    _save(model, tokenizer, meta, args, args.steps, final=True)
    print(f"\nDone. Saved to {args.out}")


def _save(model, tokenizer, meta, args, step, final=False, best=False):
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    lm  = raw.lm._orig_mod if hasattr(raw.lm, "_orig_mod") else raw.lm

    save_meta = dict(meta)
    save_meta["tokenizer"]  = tokenizer.state()
    save_meta["sft"]        = True
    save_meta["has_vision"] = True

    payload = {
        "model":       lm.state_dict(),
        "vision_proj": raw.vision_proj.state_dict(),
        "step":        step,
        "meta":        save_meta,
    }

    if best:
        path = args.out.replace(".pt", "_best.pt")
    elif final:
        path = args.out
    else:
        path = args.out.replace(".pt", f"_step{step}.pt")

    torch.save(payload, path)
    if not best:
        tqdm.write(f"  Checkpoint → {path}")

        prev_path = args.out.replace(".pt", f"_step{step - args.save_every}.pt")
        if os.path.exists(prev_path):
            os.remove(prev_path)


if __name__ == "__main__":
    main()
