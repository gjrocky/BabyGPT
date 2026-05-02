import json
import os
import sys
from unittest.mock import MagicMock
import pytest
import torch

# Mock transformers + PIL before importing vision_finetune so tests run
# without those heavy dependencies installed locally.
sys.modules.setdefault("transformers", MagicMock())
sys.modules.setdefault("PIL", MagicMock())
sys.modules.setdefault("PIL.Image", MagicMock())

from data import BPETokenizer
from model import VisionBabyGPT
from vision_finetune import tokenize_conversation, get_lr, load_data


BLOCK_SIZE = 64


@pytest.fixture
def tokenizer():
    tok = BPETokenizer()
    tok.train("User: hello Assistant: hi User: how are you Assistant: good", vocab_size=280, verbose=False)
    return tok


# ─── tokenize_conversation ────────────────────────────────────────────────────

def test_tokenize_conversation_basic(tokenizer):
    turns = [("User", "hello"), ("Assistant", "hi")]
    ids, mask = tokenize_conversation(tokenizer, turns, BLOCK_SIZE)
    assert len(ids) == len(mask)
    assert len(ids) > 0


def test_tokenize_conversation_mask_values(tokenizer):
    turns = [("User", "hello"), ("Assistant", "hi")]
    ids, mask = tokenize_conversation(tokenizer, turns, BLOCK_SIZE)
    assert set(mask).issubset({0, 1})


def test_tokenize_conversation_assistant_masked(tokenizer):
    turns = [("User", "hello"), ("Assistant", "hi there")]
    ids, mask = tokenize_conversation(tokenizer, turns, BLOCK_SIZE)
    # At least some tokens should be loss-bearing (assistant)
    assert sum(mask) > 0
    # At least some tokens should be masked out (user)
    assert sum(1 for m in mask if m == 0) > 0


def test_tokenize_conversation_truncates(tokenizer):
    long_content = "hello " * 100
    turns = [("User", long_content), ("Assistant", long_content)]
    ids, mask = tokenize_conversation(tokenizer, turns, BLOCK_SIZE)
    max_len = BLOCK_SIZE - VisionBabyGPT.NUM_IMG_TOKENS - 1
    assert len(ids) <= max_len
    assert len(mask) <= max_len


def test_tokenize_conversation_empty_turns(tokenizer):
    ids, mask = tokenize_conversation(tokenizer, [], BLOCK_SIZE)
    assert ids == []
    assert mask == []


# ─── get_lr ───────────────────────────────────────────────────────────────────

def test_get_lr_warmup_starts_at_zero():
    lr = get_lr(step=0, warmup=100, total=1000, max_lr=1e-3, min_lr=1e-4)
    assert lr == 0.0


def test_get_lr_warmup_linear():
    lr_50  = get_lr(step=50,  warmup=100, total=1000, max_lr=1e-3, min_lr=1e-4)
    lr_100 = get_lr(step=100, warmup=100, total=1000, max_lr=1e-3, min_lr=1e-4)
    assert lr_50 < lr_100


def test_get_lr_cosine_decreases():
    lr_200 = get_lr(step=200, warmup=100, total=1000, max_lr=1e-3, min_lr=1e-4)
    lr_800 = get_lr(step=800, warmup=100, total=1000, max_lr=1e-3, min_lr=1e-4)
    assert lr_200 > lr_800


def test_get_lr_ends_at_min():
    lr = get_lr(step=1000, warmup=100, total=1000, max_lr=1e-3, min_lr=1e-4)
    assert lr == pytest.approx(1e-4)


def test_get_lr_never_exceeds_max():
    for step in range(0, 1001, 100):
        lr = get_lr(step=step, warmup=100, total=1000, max_lr=1e-3, min_lr=1e-4)
        assert lr <= 1e-3 + 1e-9


# ─── load_data ────────────────────────────────────────────────────────────────

def test_load_data_filters_missing_images(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # Create one real image file
    (img_dir / "000000.jpg").write_bytes(b"fake")

    conversations = [
        {"image": "000000.jpg", "conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]},
        {"image": "missing.jpg", "conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]},
    ]
    (tmp_path / "conversations.json").write_text(json.dumps(conversations))

    data = load_data(str(tmp_path))
    assert len(data) == 1
    assert data[0]["image"] == "000000.jpg"


def test_load_data_adds_image_path(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "000000.jpg").write_bytes(b"fake")

    conversations = [
        {"image": "000000.jpg", "conversations": []},
    ]
    (tmp_path / "conversations.json").write_text(json.dumps(conversations))

    data = load_data(str(tmp_path))
    assert "image_path" in data[0]
    assert data[0]["image_path"].endswith("000000.jpg")
