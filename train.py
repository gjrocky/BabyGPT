"""
Train TinyGPT — supports both small text files and large pre-tokenized datasets.

Two modes
---------
1. Text mode (--data)
   The original mode.  Loads a single .txt file, trains a BPE tokenizer on it,
   then trains the model.  Good for experiments with small corpora.

2. Binary mode (--train-bin / --val-bin / --tokenizer)
   Production mode.  Reads pre-tokenized .bin files produced by collect/prepare.py.
   Uses numpy.memmap so the dataset never has to fit in RAM — you can train on
   hundreds of GB with 64 GB of system memory.

Key features added for serious training
-----------------------------------------
bf16 mixed precision
  Stores weights in float32 but does forward/backward in bfloat16.  bfloat16
  has the same exponent range as float32 (no overflow risk) but half the bits.
  This roughly doubles throughput on Ampere+ GPUs (RTX 3000+ / A-series) and
  halves the memory used for activations.  On your RTX 5090 expect 2–3× speedup.

torch.compile (--compile)
  PyTorch 2.0+ can compile the model to optimised GPU kernels (like XLA for JAX).
  Adds ~60s of startup time but reduces per-step time by 20–30%.  Use it for
  runs longer than a few minutes.

Cosine learning rate schedule with warmup
  Modern LLM training never uses a flat learning rate.  Instead:
    - Linear warmup: LR rises from 0 to peak over warmup_steps.
      This stabilises early training before the model has learned anything.
    - Cosine decay: LR follows a cosine curve from peak to min_lr.
      The model makes large updates early (while loss is high) and small
      updates late (fine-tuning the last few percent of quality).

Gradient accumulation (--grad-accum)
  Simulates a larger batch by accumulating gradients over multiple forward
  passes before calling optimizer.step().  If batch_size=8 and grad_accum=8,
  the effective batch is 64 — same as running batch_size=64 but using 8× less
  VRAM.  Larger effective batches generally improve training stability.

Gradient clipping (--grad-clip)
  After backprop but before the optimiser step, we scale down the gradient
  vector if its norm exceeds grad_clip (typically 1.0).  This prevents a
  single bad batch from sending weights to infinity ("gradient explosion"),
  which is common early in training on diverse data.

Periodic checkpointing (--save-every)
  Saves a checkpoint every N steps so you can resume after interruption.
  Training large models for weeks without periodic saves is a mistake.

Validation loss (--eval-every)
  Evaluates on the validation set every N steps.  Training loss always goes
  down — validation loss tells you if the model is actually generalising
  or just memorising the training data (overfitting).

Usage (text mode, quick experiment)
-----
  python train.py --data data.txt --epochs 5

Usage (binary mode, large-scale training with your RTX 5090)
-----
  python train.py \\
    --train-bin  data/train.bin \\
    --val-bin    data/val.bin   \\
    --tokenizer  data/tokenizer.json \\
    --n-embd 1024 --n-layer 24 --n-head 16 \\
    --block-size 1024 --batch-size 8 --grad-accum 8 \\
    --compile --bf16

Recommended hyperparameters for RTX 5090 (32 GB VRAM)
------------------------------------------------------
  Safe (~350M params, fits easily):
    --n-embd 1024 --n-layer 24 --n-head 16 --block-size 1024

  Ambitious (~1B params, needs grad checkpointing — add later):
    --n-embd 2048 --n-layer 24 --n-head 16 --block-size 1024
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from data import BPETokenizer, load_text
from model import TinyGPT


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float) -> float:
    """
    Linear warmup → cosine decay.

    During warmup (step < warmup_steps):
      LR rises linearly from 0 to max_lr.
      The model starts with tiny updates so the randomly-initialised weights
      don't immediately cause huge gradient explosions.

    After warmup (step ≥ warmup_steps):
      LR follows a cosine curve from max_lr down to min_lr.
      The cosine shape is smooth — no abrupt drops — and has been empirically
      found to produce better final models than step-decay schedules.
    """
    if step < warmup_steps:
        return max_lr * step / max_steps(warmup_steps, 1)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def max_steps(a: int, b: int) -> int:
    return max(a, b)


# ---------------------------------------------------------------------------
# Batch sampler — text mode
# ---------------------------------------------------------------------------

def get_batch_text(
    data:       torch.Tensor,
    block_size: int,
    batch_size: int,
    device:     str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random mini-batch from an in-memory encoded corpus.

    x (input)  = tokens at positions [i,   i + block_size)
    y (target) = tokens at positions [i+1, i + block_size + 1)

    y is x shifted one step to the right.  The model learns: for every
    position t in x, predict x[t+1].  This is called teacher forcing —
    at training time we always give the model the ground-truth context,
    never its own previous predictions.
    """
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x  = torch.stack([data[i     : i + block_size    ] for i in ix])
    y  = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Batch sampler — binary mode (numpy memmap, large datasets)
# ---------------------------------------------------------------------------

def get_batch_bin(
    data:       np.memmap,
    block_size: int,
    batch_size: int,
    device:     str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random mini-batch from a memory-mapped binary file.

    numpy.memmap maps the file into the process address space.  When we
    index into it, the OS loads only the pages we access from disk — we
    never hold the full dataset in RAM.  This makes it possible to train
    on a 200 GB dataset with 64 GB of system RAM.
    """
    ix = np.random.randint(0, len(data) - block_size - 1, size=(batch_size,), dtype=np.int64)
    x  = torch.from_numpy(np.stack([data[i     : i + block_size    ].astype(np.int64) for i in ix]))
    y  = torch.from_numpy(np.stack([data[i + 1 : i + block_size + 1].astype(np.int64) for i in ix]))
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Validation loss
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_val_loss(
    model:      TinyGPT,
    val_data,   # torch.Tensor or np.memmap
    block_size: int,
    batch_size: int,
    device:     str,
    eval_iters: int = 50,
    use_bin:    bool = False,
) -> float:
    """
    Evaluate the average loss on the validation set.

    We run eval_iters batches and average the loss.  More iters = more
    accurate estimate but more time.  50 is a reasonable default.

    model.eval() disables dropout so every forward pass is deterministic.
    torch.no_grad() skips building the computation graph, saving memory.
    """
    model.eval()
    losses = []
    get_batch = get_batch_bin if use_bin else get_batch_text
    for _ in range(eval_iters):
        xb, yb = get_batch(val_data, block_size, batch_size, device)
        _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Train TinyGPT")

    # --- Data: text mode ---
    p.add_argument("--data",          type=str, default=None,
                   help="Path to a single .txt training file (text mode)")

    # --- Data: binary mode (from collect/prepare.py) ---
    p.add_argument("--train-bin",     type=str, default=None,
                   help="Path to pre-tokenized train.bin (binary mode)")
    p.add_argument("--val-bin",       type=str, default=None,
                   help="Path to pre-tokenized val.bin (binary mode)")
    p.add_argument("--tokenizer",     type=str, default=None,
                   help="Path to tokenizer.json (binary mode)")

    # --- Output ---
    p.add_argument("--out",           type=str, default="checkpoint.pt")
    p.add_argument("--save-every",    type=int, default=1000,
                   help="Save checkpoint every N steps")
    p.add_argument("--eval-every",    type=int, default=250,
                   help="Evaluate validation loss every N steps")

    # --- Tokenizer (text mode only) ---
    p.add_argument("--bpe-vocab-size", type=int, default=32_000)

    # --- Model architecture ---
    p.add_argument("--block-size",    type=int, default=1024)
    p.add_argument("--n-layer",       type=int, default=12)
    p.add_argument("--n-head",        type=int, default=12)
    p.add_argument("--n-embd",        type=int, default=768)
    p.add_argument("--dropout",       type=float, default=0.1)

    # --- Training ---
    p.add_argument("--steps",         type=int, default=100_000,
                   help="Total training steps (binary mode)")
    p.add_argument("--epochs",        type=int, default=1,
                   help="Training epochs (text mode)")
    p.add_argument("--batch-size",    type=int, default=8)
    p.add_argument("--grad-accum",    type=int, default=8,
                   help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    p.add_argument("--lr",            type=float, default=6e-4,
                   help="Peak learning rate (AdamW)")
    p.add_argument("--min-lr",        type=float, default=6e-5,
                   help="Minimum LR at end of cosine decay (typically 0.1 × lr)")
    p.add_argument("--warmup-steps",  type=int, default=2000,
                   help="Linear warmup steps")
    p.add_argument("--grad-clip",     type=float, default=1.0,
                   help="Gradient clipping norm (0 = disabled)")
    p.add_argument("--weight-decay",  type=float, default=0.1)
    p.add_argument("--beta1",         type=float, default=0.9)
    p.add_argument("--beta2",         type=float, default=0.95,
                   help="AdamW beta2 — 0.95 (not 0.999) works better for LLMs")

    # --- Hardware ---
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bf16",          action="store_true",
                   help="Use bfloat16 mixed precision (recommended for RTX 3000+)")
    p.add_argument("--adam8bit",      action="store_true",
                   help="Use 8-bit AdamW (bitsandbytes) — cuts optimizer VRAM by ~4x, required for large models")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="Gradient checkpointing — recomputes activations to save VRAM (slower but fits larger models)")
    p.add_argument("--compile",       action="store_true",
                   help="torch.compile the model (20-30% speedup, 60s startup)")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--resume",        type=str, default=None,
                   help="Resume training from this checkpoint path")

    args = p.parse_args()

    # Validate mode
    use_bin = args.train_bin is not None
    if not use_bin and args.data is None:
        raise SystemExit("Provide either --data (text mode) or --train-bin/--val-bin (binary mode)")

    torch.manual_seed(args.seed)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    if use_bin:
        print(f"Binary mode — loading {args.train_bin}")
        train_data = np.memmap(args.train_bin, dtype=np.uint16, mode="r")
        val_data   = np.memmap(args.val_bin,   dtype=np.uint16, mode="r") if args.val_bin else None
        with open(args.tokenizer) as f:
            tokenizer = BPETokenizer.from_state(json.load(f))
        vocab_size = tokenizer.vocab_size
        print(f"  Train tokens: {len(train_data):,}  Val tokens: {len(val_data) if val_data is not None else 0:,}")
    else:
        print(f"Text mode — loading {args.data}")
        text = load_text(args.data)
        print(f"  {len(text):,} characters  |  Training BPE tokenizer...")
        tokenizer = BPETokenizer()
        tokenizer.train(text, vocab_size=args.bpe_vocab_size)
        ids        = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n_val      = max(1, len(ids) // 10)
        train_data = ids[:-n_val]
        val_data   = ids[-n_val:]
        vocab_size = tokenizer.vocab_size

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = TinyGPT(
        vocab_size              = vocab_size,
        block_size              = args.block_size,
        n_layer                 = args.n_layer,
        n_head                  = args.n_head,
        n_embd                  = args.n_embd,
        dropout                 = args.dropout,
        gradient_checkpointing  = args.grad_checkpoint,
    ).to(args.device)

    # Store weights in bf16 to halve weight memory — critical for large models.
    # bitsandbytes AdamW8bit supports bf16 parameters natively.
    if args.bf16 and torch.cuda.is_bf16_supported():
        model = model.to(torch.bfloat16)
        print("Model weights stored in bf16")

    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt       = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt.get("step", 0)

    # torch.compile wraps the model in a compiled version.  The first forward
    # pass triggers JIT compilation (~60s), then all subsequent passes are
    # faster.  Only worthwhile for runs > a few hundred steps.
    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------
    # AdamW separates weight decay from the gradient update (unlike Adam,
    # which applies it incorrectly through the gradient).  We apply weight
    # decay only to weight matrices (2D tensors), not to biases or LayerNorms.
    decay_params   = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    param_groups   = [
        {"params": decay_params,   "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    if args.adam8bit:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(
                param_groups,
                lr=args.lr,
                betas=(args.beta1, args.beta2),
            )
            print("Using 8-bit AdamW (bitsandbytes)")
        except ImportError:
            raise SystemExit("8-bit Adam requires bitsandbytes: pip install bitsandbytes")
    else:
        opt = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            fused=True if args.device == "cuda" else False,
        )
    if args.resume and "optimizer" in ckpt:
        opt.load_state_dict(ckpt["optimizer"])

    # ------------------------------------------------------------------
    # Mixed precision scaler
    # ------------------------------------------------------------------
    # bfloat16 (bf16) computes forward/backward passes in 16-bit but keeps
    # the master weights in float32.  This means:
    #   - ~2× memory savings for activations
    #   - ~2–3× faster matmuls on Ampere+ GPUs (RTX 3000+, A-series)
    #   - No overflow risk (bf16 has same exponent range as float32)
    # GradScaler is not needed for bf16 (only for fp16), but we include the
    # autocast context manager either way.
    dtype   = torch.bfloat16 if args.bf16 and torch.cuda.is_bf16_supported() else torch.float32
    scaler  = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    ctx     = torch.amp.autocast(device_type="cuda", dtype=dtype) if args.device == "cuda" else torch.nullcontext()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    if use_bin:
        total_steps = args.steps
    else:
        steps_per_epoch = max(1, len(train_data) // (args.batch_size * args.block_size * args.grad_accum))
        total_steps     = steps_per_epoch * args.epochs

    get_batch = get_batch_bin if use_bin else get_batch_text

    print(f"\nTraining  steps={total_steps:,}  device={args.device}  dtype={dtype}  "
          f"effective_batch={args.batch_size * args.grad_accum}\n")

    bar = tqdm(range(start_step, total_steps), desc="Training", unit=" steps")

    opt.zero_grad(set_to_none=True)
    accum_loss = 0.0

    for step in bar:
        t0 = time.time()

        # Gradient accumulation inner loop
        # We run grad_accum forward passes and sum their gradients before
        # doing one optimiser step.  This lets us simulate a large batch
        # without needing the VRAM to hold a large batch all at once.
        for micro_step in range(args.grad_accum):
            xb, yb = get_batch(train_data, args.block_size, args.batch_size, args.device)
            with ctx:
                _, loss = model(xb, yb)
                # Divide by grad_accum so the gradient is the average over
                # all micro-steps, not the sum.
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # Gradient clipping
        # Computes the L2 norm of the full gradient vector.  If it exceeds
        # grad_clip, scales all gradients down proportionally.  This prevents
        # a single bad batch from causing a catastrophic weight update.
        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Update learning rate according to schedule
        lr = get_lr(step, args.warmup_steps, total_steps, args.lr, args.min_lr)
        for pg in opt.param_groups:
            pg["lr"] = lr

        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        # Throughput
        dt    = time.time() - t0
        toks  = args.batch_size * args.block_size * args.grad_accum
        bar.set_postfix(loss=f"{accum_loss:.4f}", lr=f"{lr:.2e}", tok_s=f"{toks/dt:.0f}")
        accum_loss = 0.0

        # Validation
        if val_data is not None and step % args.eval_every == 0 and step > 0:
            val_loss = estimate_val_loss(
                model, val_data, args.block_size, args.batch_size,
                args.device, use_bin=use_bin,
            )
            tqdm.write(f"step {step:>7}  val_loss={val_loss:.4f}  lr={lr:.2e}")

        # Checkpoint — pass prev_step so the old checkpoint gets deleted after saving
        if step % args.save_every == 0 and step > 0:
            prev_step = step - args.save_every
            _save(model, opt, tokenizer, args, step, prev_step=prev_step)

    # Final checkpoint
    _save(model, opt, tokenizer, args, total_steps, final=True)
    print(f"\nTraining complete.  Checkpoint saved to {args.out}")


def _save(model, opt, tokenizer, args, step: int, final: bool = False, prev_step: int = 0) -> None:
    path = args.out if final else args.out.replace(".pt", f"_step{step}.pt")
    raw  = model._orig_mod if hasattr(model, "_orig_mod") else model  # unwrap compile
    torch.save({
        "model":     raw.state_dict(),
        "optimizer": opt.state_dict(),
        "step":      step,
        "meta": {
            "vocab_size":  tokenizer.vocab_size,
            "block_size":  args.block_size,
            "n_layer":     args.n_layer,
            "n_head":      args.n_head,
            "n_embd":      args.n_embd,
            "dropout":     args.dropout,
            "tokenizer":   tokenizer.state(),
        },
    }, path)
    # Delete previous step checkpoint to avoid filling disk on large models
    if not final and prev_step > 0:
        prev_path = args.out.replace(".pt", f"_step{prev_step}.pt")
        if os.path.exists(prev_path):
            os.remove(prev_path)
    if not final:
        tqdm.write(f"  Checkpoint saved → {path}")


if __name__ == "__main__":
    main()
