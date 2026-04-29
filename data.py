from __future__ import annotations

import re
from collections import defaultdict

_RE_CHUNKS = re.compile(r'\S+|\s+')


def _count_pairs(ids: list[int]) -> dict[tuple[int, int], int]:
    """Return a frequency table of every adjacent (a, b) pair in *ids*."""
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] += 1
    return counts


def _apply_merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of *pair* in *ids* with *new_id*."""
    out: list[int] = []
    i = 0
    a, b = pair
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def _merge_word(word: tuple[int, ...], a: int, b: int, new_id: int) -> tuple[int, ...]:
    out: list[int] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def load_text(path: str) -> str:
    """Read a UTF-8 text file and return the full contents as a string."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


class BPETokenizer:
    """
    Byte-level BPE tokenizer.

    Vocabulary layout:
      ids 0-255   → single raw bytes  (always present, never merged away)
      ids 256+    → merged token pairs learned during training
    """

    def __init__(self) -> None:
        # merges: maps (left_id, right_id) -> merged_id, in training order
        self.merges: dict[tuple[int, int], int] = {}
        # vocab: maps id -> the raw bytes that token represents
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int, verbose: bool = True) -> None:

        assert vocab_size >= 256, "vocab_size must be >= 256 (one slot per byte)"
        n_merges = vocab_size - 256

        # Build word-frequency dict: tuple-of-byte-ints → count
        word_freqs: dict[tuple[int, ...], int] = defaultdict(int)
        for chunk in _RE_CHUNKS.findall(text):
            word_freqs[tuple(chunk.encode("utf-8"))] += 1
        word_freqs = dict(word_freqs)
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_to_words: dict[tuple[int, int], set] = defaultdict(set)
        for word, freq in word_freqs.items():
            for j in range(len(word) - 1):
                p = (word[j], word[j + 1])
                pair_counts[p] += freq
                pair_to_words[p].add(word)

        for i in range(n_merges):
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.__getitem__)
            best_freq = pair_counts[best_pair]
            if best_freq < 2:
                break

            new_id = 256 + i
            a, b = best_pair

            # Only iterate word types that actually contain best_pair
            for word in list(pair_to_words.get(best_pair, ())):
                freq = word_freqs[word]

                # Remove this word's contribution from all its pairs
                for j in range(len(word) - 1):
                    p = (word[j], word[j + 1])
                    pair_counts[p] -= freq
                    pair_to_words[p].discard(word)

                new_word = _merge_word(word, a, b, new_id)

                # new_id is brand-new so new_word can't pre-exist in word_freqs
                word_freqs[new_word] = freq
                del word_freqs[word]

                # Register new_word's pairs
                for j in range(len(new_word) - 1):
                    p = (new_word[j], new_word[j + 1])
                    pair_counts[p] += freq
                    pair_to_words[p].add(new_word)

            pair_counts.pop(best_pair, None)
            pair_to_words.pop(best_pair, None)

            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if verbose and (i % 100 == 0 or i == n_merges - 1):
                token_str = self.vocab[new_id].decode("utf-8", errors="replace")
                print(
                    f"  merge {i+1:4d}/{n_merges}  "
                    f"pair={best_pair}  new_id={new_id}  "
                    f"token='{token_str}'  freq={best_freq}"
                )

    def encode(self, text: str) -> list[int]:
        cache: dict[bytes, list[int]] = {}
        result: list[int] = []
        for chunk in _RE_CHUNKS.findall(text):
            key = chunk.encode("utf-8")
            if key not in cache:
                cache[key] = self._encode_bytes(key)
            result.extend(cache[key])
        return result

    def _encode_bytes(self, data: bytes) -> list[int]:
        ids = list(data)
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            best = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if best not in self.merges:
                break
            ids = _apply_merge(ids, best, self.merges[best])
        return ids


    def decode(self, ids: list[int]) -> str:
        """
        Convert a list of token ids back to a string.

        Each id maps to a byte sequence; concatenating them reconstructs the
        original UTF-8 bytes.
        """
        raw_bytes = b"".join(self.vocab[i] for i in ids)
        return raw_bytes.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def state(self) -> dict:
        """Return a JSON-serialisable dict of all tokenizer state."""
        return {
            "merges": [[a, b, new_id] for (a, b), new_id in self.merges.items()],
            "vocab":  {str(k): list(v) for k, v in self.vocab.items()},
        }

    @classmethod
    def from_state(cls, data: dict) -> "BPETokenizer":
        """Reconstruct a tokenizer from a dict produced by :meth:`state`."""
        tok = cls()
        tok.merges = {(a, b): new_id for a, b, new_id in data["merges"]}
        tok.vocab  = {int(k): bytes(v) for k, v in data["vocab"].items()}
        return tok
