import pytest
import torch
from model import BabyGPT, VisionBabyGPT

VOCAB      = 256
BLOCK      = 64
N_LAYER    = 2
N_HEAD     = 2
N_EMBD     = 32
CLIP_DIM   = 16   # fake CLIP hidden size for speed
NUM_PATCHES = VisionBabyGPT.NUM_IMG_TOKENS   # 49


@pytest.fixture
def baby():
    return BabyGPT(
        vocab_size=VOCAB, block_size=BLOCK,
        n_layer=N_LAYER, n_head=N_HEAD, n_embd=N_EMBD, dropout=0.0,
    )


@pytest.fixture
def vision(baby):
    return VisionBabyGPT(baby, clip_hidden=CLIP_DIM)


# ─── BabyGPT ─────────────────────────────────────────────────────────────────

def test_babygpt_forward_logit_shape(baby):
    B, T = 2, 16
    idx  = torch.randint(0, VOCAB, (B, T))
    logits, loss = baby(idx)
    assert logits.shape == (B, T, VOCAB)
    assert loss is None


def test_babygpt_forward_with_targets(baby):
    B, T  = 2, 16
    idx   = torch.randint(0, VOCAB, (B, T))
    tgts  = torch.randint(0, VOCAB, (B, T))
    _, loss = baby(idx, tgts)
    assert loss is not None
    assert loss.ndim == 0
    assert loss.item() > 0


def test_babygpt_generate_length(baby):
    idx = torch.randint(0, VOCAB, (1, 4))
    out = baby.generate(idx, max_new_tokens=10)
    assert out.shape == (1, 14)


def test_babygpt_generate_top_k(baby):
    idx = torch.randint(0, VOCAB, (1, 4))
    out = baby.generate(idx, max_new_tokens=8, top_k=5)
    assert out.shape == (1, 12)


def test_babygpt_block_size_assertion(baby):
    idx = torch.randint(0, VOCAB, (1, BLOCK + 1))
    with pytest.raises(AssertionError):
        baby(idx)


# ─── VisionBabyGPT ───────────────────────────────────────────────────────────

def test_vision_forward_no_image(vision):
    B, T = 2, 16
    idx  = torch.randint(0, VOCAB, (B, T))
    logits, loss = vision(idx)
    assert logits.shape == (B, T, VOCAB)
    assert loss is None


def test_vision_forward_with_image(vision):
    B, T      = 2, 10
    idx       = torch.randint(0, VOCAB, (B, T))
    img_feats = torch.randn(B, NUM_PATCHES, CLIP_DIM)
    logits, loss = vision(idx, image_features=img_feats)
    # logits covers image tokens + text tokens
    assert logits.shape == (B, NUM_PATCHES + T, VOCAB)
    assert loss is None


def test_vision_loss_only_on_text_tokens(vision):
    B, T      = 2, 10
    idx       = torch.randint(0, VOCAB, (B, T))
    tgts      = torch.randint(0, VOCAB, (B, T))
    img_feats = torch.randn(B, NUM_PATCHES, CLIP_DIM)

    _, loss_vision = vision(idx, targets=tgts, image_features=img_feats)
    _, loss_text   = vision(idx, targets=tgts)

    # Both should produce a valid scalar loss
    assert loss_vision is not None and loss_vision.ndim == 0
    assert loss_text   is not None and loss_text.ndim   == 0


def test_vision_generate_no_image(vision):
    idx = torch.randint(0, VOCAB, (1, 4))
    out = vision.generate(idx, max_new_tokens=6)
    assert out.shape == (1, 10)


def test_vision_generate_with_image(vision):
    idx       = torch.randint(0, VOCAB, (1, 4))
    img_feats = torch.randn(1, NUM_PATCHES, CLIP_DIM)
    out = vision.generate(idx, max_new_tokens=6, image_features=img_feats)
    assert out.shape == (1, 10)


def test_vision_proj_shape(vision):
    img_feats  = torch.randn(2, NUM_PATCHES, CLIP_DIM)
    projected  = vision.vision_proj(img_feats)
    assert projected.shape == (2, NUM_PATCHES, N_EMBD)


def test_vision_generate_respects_block_size(vision):
    # Text tokens should be capped at block_size - NUM_PATCHES
    max_text = BLOCK - NUM_PATCHES
    idx       = torch.randint(0, VOCAB, (1, max_text))
    img_feats = torch.randn(1, NUM_PATCHES, CLIP_DIM)
    # Should not raise
    out = vision.generate(idx, max_new_tokens=2, image_features=img_feats)
    assert out.shape[1] == max_text + 2
