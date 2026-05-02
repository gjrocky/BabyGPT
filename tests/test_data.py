import pytest
from data import BPETokenizer, load_text


# ─── BPETokenizer ─────────────────────────────────────────────────────────────

@pytest.fixture
def trained_tok():
    tok  = BPETokenizer()
    text = "hello world hello world hello world the quick brown fox"
    tok.train(text, vocab_size=280, verbose=False)
    return tok


def test_initial_vocab_is_256_bytes():
    tok = BPETokenizer()
    assert tok.vocab_size == 256
    assert all(tok.vocab[i] == bytes([i]) for i in range(256))


def test_train_grows_vocab(trained_tok):
    assert trained_tok.vocab_size > 256


def test_encode_returns_list_of_ints(trained_tok):
    ids = trained_tok.encode("hello")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_roundtrip(trained_tok):
    text = "hello world"
    assert trained_tok.decode(trained_tok.encode(text)) == text


def test_encode_never_unknown():
    tok = BPETokenizer()
    ids = tok.encode("任意の UTF-8 テキスト 🎉")
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)


def test_empty_string(trained_tok):
    assert trained_tok.encode("") == []
    assert trained_tok.decode([]) == ""


def test_state_roundtrip(trained_tok):
    state = trained_tok.state()
    tok2  = BPETokenizer.from_state(state)
    text  = "hello world"
    assert tok2.encode(text) == trained_tok.encode(text)
    assert tok2.vocab_size   == trained_tok.vocab_size


def test_merges_reduce_token_count(trained_tok):
    base = BPETokenizer()
    text = "hello world"
    assert len(trained_tok.encode(text)) <= len(base.encode(text))


def test_vocab_size_property(trained_tok):
    assert trained_tok.vocab_size == len(trained_tok.vocab)
