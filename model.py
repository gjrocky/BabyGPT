from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head   = n_head
        self.head_dim = n_embd // n_head
        self.dropout  = dropout

        self.qkv  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.proj._is_residual = True

        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p = self.dropout if self.training else 0.0,
            is_causal = True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1  = nn.Linear(n_embd, 4 * n_embd)
        self.fc2  = nn.Linear(4 * n_embd, n_embd)
        self.drop = nn.Dropout(dropout)
        self.fc2._is_residual = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2  = nn.LayerNorm(n_embd)
        self.mlp  = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class BabyGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 1024,
        n_layer:    int = 12,
        n_head:     int = 12,
        n_embd:     int = 768,
        dropout:    float = 0.1,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.block_size = block_size

        self.tok_emb  = nn.Embedding(vocab_size, n_embd)
        self.pos_emb  = nn.Embedding(block_size, n_embd)
        self.drop     = nn.Dropout(dropout)
        self.blocks   = nn.ModuleList(Block(n_embd, n_head, dropout) for _ in range(n_layer))
        self.ln_f     = nn.LayerNorm(n_embd)
        self.head     = nn.Linear(n_embd, vocab_size, bias=False)
        self.gradient_checkpointing = gradient_checkpointing

        self.tok_emb.weight = self.head.weight
        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"BabyGPT  params={n_params:,}  vocab={vocab_size}  "
              f"ctx={block_size}  layers={n_layer}  heads={n_head}  embd={n_embd}")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "_is_residual", False):
                std *= (2 * len(self.blocks)) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.block_size

        pos  = torch.arange(T, device=idx.device)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x      = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx:            torch.Tensor,
        max_new_tokens: int,
        temperature:    float = 1.0,
        top_k:          int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond  = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits    = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat((idx, idx_next), dim=1)

        return idx


class VisionBabyGPT(nn.Module):

    NUM_IMG_TOKENS = 49  # ViT-B/32: (224/32)^2 = 49 patch tokens

    def __init__(self, lm: BabyGPT, clip_hidden: int = 768) -> None:
        super().__init__()
        self.lm        = lm
        self.vision_proj = nn.Linear(clip_hidden, lm.tok_emb.embedding_dim)

    def forward(
        self,
        idx:            torch.Tensor,
        targets:        torch.Tensor | None = None,
        image_features: torch.Tensor | None = None,
    ):
        B, T = idx.shape
        pos      = torch.arange(T, device=idx.device)
        text_emb = self.lm.drop(self.lm.tok_emb(idx) + self.lm.pos_emb(pos))

        if image_features is not None:
            img_emb = self.vision_proj(image_features)   # (B, 49, n_embd)
            x       = torch.cat([img_emb, text_emb], dim=1)
            num_img = img_emb.shape[1]
        else:
            x       = text_emb
            num_img = 0

        for block in self.lm.blocks:
            if self.lm.gradient_checkpointing and self.lm.training:
                x = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x      = self.lm.ln_f(x)
        logits = self.lm.head(x)

        loss = None
        if targets is not None:
            text_logits = logits[:, num_img:, :]
            loss = F.cross_entropy(
                text_logits.view(-1, text_logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx:            torch.Tensor,
        max_new_tokens: int,
        temperature:    float = 1.0,
        top_k:          int | None = None,
        image_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_img  = self.NUM_IMG_TOKENS if image_features is not None else 0
        max_text = self.lm.block_size - num_img

        for _ in range(max_new_tokens):
            idx_cond  = idx[:, -max_text:]
            logits, _ = self(idx_cond, image_features=image_features)
            logits    = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat((idx, idx_next), dim=1)

        return idx
