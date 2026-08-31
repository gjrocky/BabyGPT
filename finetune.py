"""
Supervised Fine-Tuning (SFT) — turn a pretrained TinyGPT into a chatbot.

Key difference from pretraining
---------------------------------
Loss is computed ONLY on assistant turns.  User turns are masked out by
setting their target tokens to -100, which F.cross_entropy ignores.
This teaches the model to generate responses rather than predict any token.

Expected data format (produced by collect/sharegpt.py and collect/openassistant.py):

  User: Hello, how are you?

  Assistant: I'm doing well! How can I help?

  User: Can you explain what BPE tokenization is?

  Assistant: Sure! BPE stands for...

Conversations within a file are separated by lines containing only ---.

Usage
-----
  # Basic: fine-tune a pretrained checkpoint
  python finetune.py --checkpoint checkpoint.pt --data raw/

  # Multiple data directories, more steps
  python finetune.py \\
    --checkpoint checkpoint.pt \\
    --data raw/sharegpt raw/openassistant \\
    --steps 10000 --lr 1e-4 --batch-size 4 --grad-accum 8

  # With GPU acceleration
  python finetune.py --checkpoint checkpoint.pt --data raw/ --bf16 --compile
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import BPETokenizer
from model import BabyGPT


# ---------------------------------------------------------------------------
# Conversation parsing
# ---------------------------------------------------------------------------

# Matches "User: ", "Assistant: ", "System: " at the start of a block or
# after a blank line — these are the turn delimiters the collect scripts use.
_TURN_RE = re.compile(r'(?:^|\n\n)(User|Assistant|System): ', re.MULTILINE)
_CONV_SEP = '---'


def parse_turns(text: str) -> list[tuple[str, str]]:
    """Extract (role, content) pairs from one conversation block."""
    matches = list(_TURN_RE.finditer(text))
    if not matches:
        return []
    turns: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        role  = m.group(1)
        start = m.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            turns.append((role, content))
    return turns


def load_conversations(dirs: list[str]) -> list[list[tuple[str, str]]]:
    """Walk directories, parse all .txt files into conversation lists."""
    paths: list[str] = []
    for d in dirs:
        paths.extend(glob.glob(os.path.join(d, '**', '*.txt'), recursive=True))
    random.shuffle(paths)
    print(f"  Found {len(paths):,} .txt files")

    convs: list[list[tuple[str, str]]] = []
    for path in tqdm(paths, desc='Parsing files', unit=' file'):
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        for block in text.split(_CONV_SEP):
            block = block.strip()
            if not block:
                continue
            turns = parse_turns(block)
            roles = {r for r, _ in turns}
            if 'User' in roles and 'Assistant' in roles:
                convs.append(turns)
    return convs


# ---------------------------------------------------------------------------
# Tokenization with loss mask
# ---------------------------------------------------------------------------

def tokenize_conversation(
    tokenizer: BPETokenizer,
    turns: list[tuple[str, str]],
) -> tuple[list[int], list[int]]:
    """
    Tokenize a conversation and produce a per-token loss mask.

    mask[i] = 1  → token i is part of an assistant turn (compute loss)
    mask[i] = 0  → token i is a user/system turn (skip loss)

    Since encode() is chunk-based (splits on whitespace boundaries), encoding
    the prefix and content separately and concatenating is identical to encoding
    the combined string.  This makes per-segment masking exact.
    """
    ids:  list[int] = []
    mask: list[int] = []

    for role, content in turns:
        prefix     = f'{role}: '
        suffix     = '\n\n'
        prefix_ids = tokenizer.encode(prefix)
        content_ids = tokenizer.encode(content)
        suffix_ids = tokenizer.encode(suffix)
        turn_ids   = prefix_ids + content_ids + suffix_ids
        ids.extend(turn_ids)
        # Compute loss on all tokens in assistant turns (prefix + content + suffix)
        mask.extend([1 if role == 'Assistant' else 0] * len(turn_ids))

    return ids, mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SFTDataset:
    """
    Packs all tokenized conversations end-to-end, then samples random
    block_size windows for training — same memory layout as pretraining,
    but now accompanied by a per-token loss mask.
    """

    def __init__(
        self,
        conversations: list[list[tuple[str, str]]],
        tokenizer: BPETokenizer,
        block_size: int,
        label: str = '',
    ) -> None:
        all_ids:  list[int] = []
        all_masks: list[int] = []

        for turns in conversations:
            ids, mask = tokenize_conversation(tokenizer, turns)
            all_ids.extend(ids)
            all_masks.extend(mask)

        self.ids       = np.array(all_ids,   dtype=np.uint16)
        self.masks     = np.array(all_masks, dtype=np.uint8)
        self.block_size = block_size

        pct = 100 * self.masks.sum() / max(len(self.masks), 1)
        print(
            f'  {label}  {len(conversations):,} conversations  '
            f'{len(self.ids):,} tokens  '
            f'({self.masks.sum():,} assistant tokens = {pct:.1f}% loss-bearing)'
        )

    def get_batch(
        self, batch_size: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ix = np.random.randint(0, len(self.ids) - self.block_size - 1, (batch_size,))
        x = torch.from_numpy(
            np.stack([self.ids [i    : i + self.block_size    ].astype(np.int64) for i in ix])
        ).to(device)
        y = torch.from_numpy(
            np.stack([self.ids [i + 1: i + self.block_size + 1].astype(np.int64) for i in ix])
        ).to(device)
        m = torch.from_numpy(
            np.stack([self.masks[i + 1: i + self.block_size + 1].astype(np.float32) for i in ix])
        ).to(device)
        return x, y, m


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def sft_loss(
    model:  BabyGPT,
    x:      torch.Tensor,
    y:      torch.Tensor,
    mask:   torch.Tensor,
    ctx,
) -> torch.Tensor:
    """
    Forward pass with assistant-only cross-entropy.

    Targets where mask == 0 are set to -100, which F.cross_entropy skips
    via ignore_index.  The loss is the mean over assistant tokens only.
    """
    with ctx:
        logits, _ = model(x)                          # (B, T, V)
    y_masked = y.masked_fill(mask == 0, -100)         # zero-out user targets
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y_masked.view(-1),
        ignore_index=-100,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_val_loss(
    model:      BabyGPT,
    val_ds:     SFTDataset,
    batch_size: int,
    device:     str,
    ctx,
    eval_iters: int = 50,
) -> float:
    model.eval()
    losses = [
        sft_loss(model, *val_ds.get_batch(batch_size, device), ctx).item()
        for _ in range(eval_iters)
    ]
    model.train()
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# LR schedule (same cosine warmup as train.py)
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup: int, total: int, max_lr: float, min_lr: float) -> float:
    if step < warmup:
        return max_lr * step / max(warmup, 1)
    if step >= total:
        return min_lr
    progress = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _save(
    model,
    opt,
    tokenizer: BPETokenizer,
    base_meta: dict,
    args,
    step: int,
    final: bool = False,
    prev_step: int = 0,
) -> None:
    path = args.out if final else args.out.replace('.pt', f'_step{step}.pt')
    raw  = model._orig_mod if hasattr(model, '_orig_mod') else model
    meta = dict(base_meta)
    meta['tokenizer'] = tokenizer.state()
    meta['sft']       = True      # signals chat.py to use chat formatting
    torch.save({'model': raw.state_dict(), 'optimizer': opt.state_dict(),
                'step': step, 'meta': meta}, path)
    # Delete previous step checkpoint to avoid filling disk
    if not final and prev_step > 0:
        prev_path = args.out.replace('.pt', f'_step{prev_step}.pt')
        if os.path.exists(prev_path):
            os.remove(prev_path)
    if not final:
        tqdm.write(f'  Checkpoint → {path}')

def main() -> None:
    p = argparse.ArgumentParser(description='SFT fine-tune TinyGPT on conversation data')
    p.add_argument('--checkpoint', required=True,
                   help='Pretrained checkpoint to fine-tune (.pt from train.py)')
    p.add_argument('--data', nargs='+', default=['raw'],
                   help='Directories containing conversation .txt files')
    p.add_argument('--out',          type=str,   default='sft_checkpoint.pt')
    p.add_argument('--val-split',    type=float, default=0.05,
                   help='The fraction of conversations held out for validation (default 0.05)')
    p.add_argument('--steps',        type=int,   default=10_000)
    p.add_argument('--batch-size',   type=int,   default=4)
    p.add_argument('--grad-accum',   type=int,   default=8)
    p.add_argument('--lr',           type=float, default=1e-4,
                   help='Highest Learning Rate: lower than pretraining (default 1e-4)')
    p.add_argument('--min-lr',       type=float, default=1e-5)
    p.add_argument('--warmup-steps', type=int,   default=200)
    p.add_argument('--grad-clip',    type=float, default=1.0)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--save-every',   type=int,   default=1000)
    p.add_argument('--eval-every',   type=int,   default=200)
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--bf16',     action='store_true',
                   help='bfloat16 mixed precision (RTX 3000+)')
    p.add_argument('--adam8bit',       action='store_true',
                   help='8-bit AdamW (bitsandbytes) — cuts optimizer VRAM by ~4x')
    p.add_argument('--grad-checkpoint', action='store_true',
                   help='Gradient checkpointing — saves VRAM at cost of speed')
    p.add_argument('--compile',        action='store_true',
                   help='torch.compile for faster training')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f'Loading pretrained checkpoint: {args.checkpoint}')
    ckpt      = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    meta      = ckpt['meta']
    tokenizer = BPETokenizer.from_state(meta['tokenizer'])
    block_size = meta['block_size']

    model = BabyGPT(
        vocab_size = meta['vocab_size'],
        block_size = block_size,
        n_layer                = meta['n_layer'],
        n_head                 = meta['n_head'],
        n_embd                 = meta['n_embd'],
        dropout                = meta.get('dropout', 0.0),
        gradient_checkpointing = args.grad_checkpoint,
    ).to(args.device)
    model.load_state_dict(ckpt['model'])

    if args.bf16 and torch.cuda.is_bf16_supported():
        model = model.to(torch.bfloat16)
        print('Model weights stored in bf16')

    if args.compile:
        print('Compiling model...')
        model = torch.compile(model)

    print(f'\nLoading conversations from: {args.data}')
    convs = load_conversations(args.data)
    print(f'  Parsed {len(convs):,} valid conversations')
    if not convs:
        raise SystemExit('No conversations found.  Check --data points to the right directory.')

    random.shuffle(convs)
    n_val       = max(1, int(len(convs) * args.val_split))
    val_convs   = convs[:n_val]
    train_convs = convs[n_val:]

    print('\nBuilding datasets:')
    train_ds = SFTDataset(train_convs, tokenizer, block_size, label='train')
    val_ds   = SFTDataset(val_convs,   tokenizer, block_size, label='val  ')

    decay_params   = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    param_groups   = [{'params': decay_params,   'weight_decay': args.weight_decay},
                      {'params': nodecay_params, 'weight_decay': 0.0}]
    if args.adam8bit:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(param_groups, lr=args.lr, betas=(0.9, 0.95))
            print('Using 8-bit AdamW (bitsandbytes)')
        except ImportError:
            raise SystemExit('8-bit Adam requires bitsandbytes: pip install bitsandbytes')
    else:
        opt = torch.optim.AdamW(
            param_groups, lr=args.lr, betas=(0.9, 0.95),
            fused=True if args.device == 'cuda' else False,
        )

    dtype  = torch.bfloat16 if args.bf16 and torch.cuda.is_bf16_supported() else torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    ctx    = (torch.amp.autocast(device_type='cuda', dtype=dtype)
              if args.device == 'cuda' else torch.nullcontext())

    print(f'\nFine-tuning  steps={args.steps:,}  device={args.device}  '
          f'dtype={dtype}  eff_batch={args.batch_size * args.grad_accum}\n')

    bar           = tqdm(range(args.steps), desc='SFT', unit=' steps')
    accum_loss    = 0.0
    best_val_loss = float('inf')
    opt.zero_grad(set_to_none=True)

    for step in bar:
        t0 = time.time()

        for _ in range(args.grad_accum):
            x, y, m = train_ds.get_batch(args.batch_size, args.device)
            loss = sft_loss(model, x, y, m, ctx) / args.grad_accum
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        lr = get_lr(step, args.warmup_steps, args.steps, args.lr, args.min_lr)
        for pg in opt.param_groups:
            pg['lr'] = lr

        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        dt   = time.time() - t0
        toks = args.batch_size * block_size * args.grad_accum
        bar.set_postfix(loss=f'{accum_loss:.4f}', lr=f'{lr:.2e}', tok_s=f'{toks/dt:.0f}')
        accum_loss = 0.0

        if step % args.eval_every == 0 and step > 0:
            val_loss = estimate_val_loss(model, val_ds, args.batch_size, args.device, ctx)
            tqdm.write(f'step {step:>7}  val_loss={val_loss:.4f}  lr={lr:.2e}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = args.out.replace('.pt', '_best.pt')
                raw = model._orig_mod if hasattr(model, '_orig_mod') else model
                best_meta = dict(meta)
                best_meta['tokenizer'] = tokenizer.state()
                best_meta['sft'] = True
                torch.save({'model': raw.state_dict(), 'step': step, 'meta': best_meta}, best_path)
                tqdm.write(f'  ★ New best  val_loss={val_loss:.4f}  → {best_path}')

        if step % args.save_every == 0 and step > 0:
            prev_step = step - args.save_every
            _save(model, opt, tokenizer, meta, args, step, prev_step=prev_step)

    _save(model, opt, tokenizer, meta, args, args.steps, final=True)
    print(f'\nFine-tuning complete.  Saved to {args.out}')


if __name__ == '__main__':
    main()
