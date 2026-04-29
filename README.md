# BabyGPT
My own LLM built entirely from scratch (In Effort to spread knowledge on LLM development)


# LLM From Scratch

A small but real language model built entirely from scratch using Python and PyTorch — no Hugging Face, no pre-trained weights, no shortcuts.

Feed it any `.txt` file.  It learns the statistical patterns in that text and generates new text in the same style.

The architecture is a **decoder-only Transformer** — the same fundamental design behind GPT-2, GPT-3, and every modern LLM.  The tokenizer is **Byte-Pair Encoding (BPE)** — the same algorithm used by GPT-2 and GPT-4.  Small enough to train on a laptop in minutes.

---

## Quick Start

```bash
pip install -r requirements.txt
```

Download training data (Tiny Shakespeare is a great start):
```
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```
Save it as `data.txt` in this folder, then:

```bash
python train.py --data data.txt --epochs 5
python chat.py --checkpoint checkpoint.pt
```

Use `--device cpu` if you have no GPU. More epochs = better output.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Primary language |
| **PyTorch** | Tensors, automatic gradients, GPU acceleration |
| **tqdm** | Progress bars during training |

---

## How It All Fits Together

```
data.txt  ──►  data.py  ──►  train.py  ──►  checkpoint.pt
                                                   │
                                              chat.py  ──►  Your terminal
```

1. `data.py` trains a BPE tokenizer that maps text ↔ integer token ids
2. `train.py` encodes the corpus and teaches the model to predict the next token
3. Trained weights + tokenizer are saved to `checkpoint.pt`
4. `chat.py` loads the checkpoint and lets you type prompts

---

## What This Model Actually Is

This is **not** a chatbot in the traditional sense — it is a **next-token predictor**.  Its one job is: given some text, predict what token comes next.

When you type a prompt into `chat.py`, it doesn't "understand" your question and formulate an answer.  It continues your text in the style of whatever it was trained on.  Train it on Shakespeare and type `"To be or"` — it will likely continue with `"not to be"`, not because it reasoned about it, but because that pattern dominated the training data.

### Base LM vs Chatbot

| | This model (base LM) | A chatbot (e.g. ChatGPT) |
|---|---|---|
| **What it does** | Continues text | Responds to instructions |
| **Training** | Next-token prediction on raw text | Same base, then fine-tuned on dialogue |
| **Input** | A prompt to continue | A user message |
| **Output** | More text in the same style | An answer/response |

ChatGPT, Claude, and Llama all **started** as base language models trained exactly like this one.  The chatbot behavior comes from a second stage called **instruction tuning**.

### What You Can Train It On

Since it's a general text predictor, it works on anything:

- **Shakespeare** → generates Shakespearean prose
- **Your own notes** → mimics your writing style
- **Source code** → generates code in the same language/style
- **Song lyrics** → generates new lyrics in that artist's style
- **Dialogue-formatted text** → behaves like a basic chatbot

---

## File-by-File Breakdown

---

### `data.py` — The BPE Tokenizer

Converts raw text into numbers the model can work with, and back again.

Neural networks don't read text — they read integers.  BPE (Byte-Pair Encoding) is the algorithm that decides which sequences of characters get their own integer id.

#### Why BPE instead of character-level?

Character-level tokenization assigns one id per character (`h`, `e`, `l`, `l`, `o` = 5 tokens).  BPE merges frequent pairs until common words and subwords get single ids (`hello` = 1 token).

This matters for two reasons:
1. **Compression** — fewer tokens means the context window covers more actual text.  128 BPE tokens ≈ 300–500 characters of English prose.
2. **Richer signal per step** — the model predicts `hello` as one unit rather than predicting each letter individually, which is a much harder and less semantically meaningful task.

GPT-2 uses BPE with a 50,257-token vocabulary.  This project trains one from scratch.

#### How BPE training works (step by step)

```
Initial state:  "hello world" → [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
                                 (one integer per byte, 0-255)

Step 1: count all adjacent pairs
        (108, 108) appears most often → merge it into id 256
        vocab now: {... 256: b'll' ...}

Step 2: count again on the merged sequence
        next most frequent pair → merge into 257
        ...repeat until vocab reaches target size
```

#### `BPETokenizer` class

| Method | What it does |
|---|---|
| `train(text, vocab_size)` | Counts pair frequencies, merges the most common pair into a new token, repeats until `vocab_size` is reached |
| `encode(text)` | Converts a string to bytes, then greedily applies the learned merges (highest-priority first) until no more apply |
| `decode(ids)` | Maps each id back to its byte sequence, concatenates, decodes as UTF-8 |
| `state()` | Returns a JSON-serialisable dict of all merges and vocab entries (saved inside the checkpoint) |
| `from_state(data)` | Reconstructs a tokenizer from a dict produced by `state()` |

**Vocabulary layout:**

| Id range | What it holds |
|---|---|
| 0 – 255 | Raw bytes (always present; these are the "base" tokens) |
| 256+ | Merged tokens learned during training |

Because every possible byte is already in the vocabulary, the encoder can **never** encounter an unknown token.  Any valid UTF-8 text can be encoded.

---

### `model.py` — The Neural Network (BabyGPT)

Defines the Transformer architecture.  Four classes that stack on top of each other.

---

#### `CausalSelfAttention` — The "Reading" Mechanism

The heart of the Transformer.  Lets the model learn which previous tokens are most relevant when predicting the next one.

**Q, K, V — Queries, Keys, Values:**

Think of it like a search engine.  For each token position:
- **Query (Q):** "What am I looking for?"
- **Key (K):**   "What do I offer?"
- **Value (V):** "What information do I contain?"

The dot product `Q × Kᵀ` scores how relevant each past position is.  Dividing by `√head_dim` keeps the values numerically stable (prevents softmax from saturating).  `softmax` turns scores into probabilities (an "attention map").  Finally `att × V` collects the weighted sum of information from all past positions.

**Multiple heads:**

Instead of one attention computation, `n_head` independent heads run in parallel, each working on a different slice of the embedding.  Different heads can specialise — one might track which noun a pronoun refers to, another might focus on punctuation structure.

**The causal mask (`torch.tril`):**

A lower-triangular matrix that enforces: position N can only attend to positions 0 through N — never the future.  Without this, the model could "cheat" by reading the answer while being trained to predict it.

```
Position 0 can see: [0]
Position 1 can see: [0, 1]
Position 2 can see: [0, 1, 2]
Position 3 can see: [0, 1, 2, 3]
```

This is what makes the model a *predictor* rather than a copier.

---

#### `MLP` — The "Thinking" Layer

After attention gathers context (which past tokens are relevant), the MLP processes *what to do* with that context:

1. Expand from `n_embd` → `4 × n_embd` (room to compute)
2. Apply **GELU** activation — a smooth non-linearity.  Without non-linearity, stacking layers is mathematically equivalent to a single linear layer, so depth would add nothing.
3. Compress back to `n_embd`

---

#### `Block` — One Transformer Layer

One complete Transformer layer = Attention + MLP, each with two key additions:

```python
x = x + self.attn(self.ln1(x))   # attention sub-layer
x = x + self.mlp(self.ln2(x))    # MLP sub-layer
```

- **LayerNorm** — normalises the values entering each sub-layer so they don't grow uncontrollably large or shrink toward zero.  Applied *before* each sub-layer (Pre-LN), which is more numerically stable than the original paper's post-LN arrangement.

- **Residual connection** (`x = x + ...`) — the sub-layer's output is *added to* the input rather than replacing it.  This means the model only needs to learn the *difference* from the current representation.  More importantly, gradients can flow directly from the output back to early layers without passing through every transformation, which is what makes training deep networks stable.

---

#### `BabyGPT` — The Full Model

The top-level class that wires everything together.

**Parameters:**

| Parameter | Default | What it controls |
|---|---|---|
| `vocab_size` | auto | Number of unique tokens (set by the tokenizer) |
| `block_size` | 128 | Max tokens the model sees at once ("context window") |
| `n_layer` | 4 | Number of stacked Transformer blocks (depth) |
| `n_head` | 4 | Number of attention heads per block |
| `n_embd` | 128 | Embedding dimension — how rich each token's representation is |

**Components:**

- `tok_emb` — Embedding table: turns each integer token id into a dense `n_embd`-dimensional vector.  Think of it as a lookup table that assigns every token a learnable "personality" described by 128 numbers.
- `pos_emb` — Positional embeddings: attention has no built-in sense of order, so a learned vector for each position (0 → block_size-1) is added so the model knows *where* each token sits.
- `blocks` — Stack of `n_layer` Transformer blocks.
- `ln_f` — Final LayerNorm after all blocks.
- `head` — Linear layer projecting `n_embd` → `vocab_size`, producing raw scores (**logits**) for every possible next token.

**Forward pass:**

1. Look up token embeddings + add position embeddings
2. Pass through all Transformer blocks sequentially
3. Final LayerNorm
4. Linear head → logits over vocabulary
5. If targets provided (training): compute cross-entropy loss

**Generation:**

1. Feed the current context (capped at `block_size` tokens)
2. Get logits for the last position only (the "next token" prediction)
3. Divide by `temperature`:
   - `< 1.0` → sharper distribution → model picks high-probability tokens → repetitive but coherent
   - `> 1.0` → flatter distribution → more random sampling → creative but less coherent
4. `softmax` → probability distribution over all tokens
5. Sample one token id (`torch.multinomial`)
6. Append to sequence and repeat

---

### `train.py` — The Training Loop

Teaches the model by showing it millions of overlapping examples from the training text.

**`get_batch()`** assembles one mini-batch:
- Picks `batch_size` random starting positions in the encoded token sequence
- `x` = input:  tokens at `[i, i + block_size)`
- `y` = target: tokens at `[i+1, i + block_size + 1)` — shifted one position right

At every position, the model is asked: *"given what you've seen so far, what comes next?"*  This is asked for all 128 positions in a sequence simultaneously.

**Each training step:**

```
1. get_batch()          → sample random examples from the corpus
2. model(xb, yb)        → forward pass; compute cross-entropy loss
3. opt.zero_grad()      → clear gradients from the previous step
4. loss.backward()      → backpropagation: compute how much each weight contributed to the error
5. opt.step()           → update every weight: w ← w − lr × gradient
```

**Why cross-entropy loss?**
The model outputs a probability distribution over the vocabulary.  Cross-entropy measures how different that distribution is from the "correct" distribution (probability 1.0 on the actual next token, 0.0 everywhere else).  Minimising it forces the model to assign high probability to the correct next token.

**Why AdamW?**
AdamW is a variant of gradient descent that:
1. Adapts the learning rate per parameter based on the history of gradients (parameters with noisy gradients get smaller updates).
2. Applies **weight decay** — a regularisation penalty that nudges weights toward zero, discouraging the model from memorising rare patterns and encouraging generalisation.

**Two-phase training:**

| Phase | What happens |
|---|---|
| BPE tokenizer training | Processes the raw text *once* to learn merges and build vocabulary |
| Model training | Runs for `--epochs` passes over the encoded token sequence |

**Checkpoint format (`checkpoint.pt`):**

```python
{
    "model": { ... pytorch weight tensors ... },
    "meta": {
        "vocab_size":  1000,
        "block_size":  128,
        "n_layer":     4,
        "n_head":      4,
        "n_embd":      128,
        "tokenizer": {
            "merges": [...],   # ordered list of BPE merge rules
            "vocab":  {...},   # id → bytes mapping
        }
    }
}
```

Everything needed to reproduce the model and tokenizer is in one file.

---

### `chat.py` — The Interface

Loads a trained checkpoint and lets you type prompts.

1. Load `checkpoint.pt` — pulls out model weights and tokenizer state
2. Reconstruct the tokenizer from `meta["tokenizer"]`
3. Rebuild the exact model architecture and load saved weights
4. Set `model.eval()` — disables training-only behaviors (like dropout)
5. Read your prompt, encode it to token ids using the BPE tokenizer
6. Call `model.generate()` to produce `max_new_tokens` new tokens
7. Decode output back to text and print it

**Flags:**

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | `checkpoint.pt` | Path to saved model |
| `--max_new_tokens` | 200 | How many tokens to generate |
| `--temperature` | 0.8 | Randomness (0.5 = focused, 1.5 = wild) |

---

## Architecture Diagram

```
Input text: "To be or"
         │
         ▼
  BPE tokenizer (data.py)
         │
         ▼
  [312, 28, 94, ...]         ← token ids, not individual characters
         │
         ▼
  Token Embeddings            (n_embd-dim vector per token)
         +
  Position Embeddings         (n_embd-dim vector per position)
         │
         ▼
  ┌──────────────────────┐
  │   Transformer Block  │  × n_layer (default: 4)
  │  ┌────────────────┐  │
  │  │  LayerNorm     │  │
  │  │  Self-Attention│  │  ← which past tokens matter?
  │  │  + residual    │  │
  │  │  LayerNorm     │  │
  │  │  MLP           │  │  ← process the gathered context
  │  │  + residual    │  │
  │  └────────────────┘  │
  └──────────────────────┘
         │
         ▼
   Final LayerNorm
         │
         ▼
   Linear → vocab_size logits
         │
         ▼
   Softmax + temperature → probabilities
         │
         ▼
   Sample → next token id
         │
         ▼
   BPE decode → text
         │
         ▼
   Repeat until done
```

---

## Default Model Size

| Metric | Value |
|---|---|
| BPE vocabulary | 1,000 tokens |
| Context window | 128 tokens (~400 characters) |
| Embedding size | 128 numbers per token |
| Transformer layers | 4 |
| Attention heads | 4 |
| Total parameters | ~1–2M |

For comparison: GPT-2 small = 117M params, 50,257-token vocab.  Starting small is intentional — you can train in minutes, see results fast, and understand exactly what each change does.

---

## How to Improve Quality

Quality is a product of three things:

1. **Data** — More text = better.  Varied, high-quality text = better.  The model will sound like whatever it is trained on.
2. **Tokenizer** — Larger `--bpe_vocab_size` means richer tokens and better compression, but takes longer to train and requires more model capacity to be useful.
3. **Model size** — Increase `--n_embd`, `--n_layer`, `--n_head` for more capacity.  More parameters = more patterns the model can store.
4. **Training time** — More epochs means more gradient steps.  Watch the loss fall — lower loss = better predictions.  Starting around ~7.0 (BPE), a well-trained small model should reach ~3.0–4.0.

```bash
# Bigger vocabulary, bigger model, longer training
python train.py --data data.txt --epochs 20 --bpe_vocab_size 2000 --n_embd 256 --n_layer 6 --n_head 8
```

---

## Next Steps (How This Grows Into a Real LLM)

| Step | Status | What it adds |
|---|---|---|
| **BPE Tokenization** | ✅ Done | Subword vocab — more efficient, richer tokens |
| **Larger dataset** | Pending | Billions of tokens (Common Crawl, books, Wikipedia) |
| **Bigger model** | Pending | More layers, wider embeddings, more heads |
| **Instruction tuning** | Pending | Fine-tune on `(question → answer)` pairs — this is what makes a chatbot |
| **RLHF** | Pending | Reinforcement learning from human feedback — aligns output with human preferences |

The architecture in `model.py` is the same foundation all the way up.  Scale and data are the only real differences between this and GPT.



---

## The Full Training Pipeline

```
collect/sharegpt.py  ──►  raw/*.txt
                               │
                          collect/prepare.py
                               │
                    data/train.bin + data/val.bin
                               │
                           train.py
                               │
                         checkpoint.pt
                               │
                          finetune.py  ◄──  raw/ (conversation data)
                               │
                       sft_checkpoint.pt
                               │
                           app.py / chat.py
```

---

### Step 1 — Data Collection (`collect/sharegpt.py`)

Downloads datasets from HuggingFace and saves them as `.txt` files into `raw/`. Each file contains conversations separated by `---`. Supports multiple dataset formats out of the box.

| Function | What it does |
|---|---|
| `format_messages` | Parses OpenAI-style `{role, content}` format (UltraChat) |
| `format_sharegpt` | Parses ShareGPT-style `{from, value}` format (Vicuna) |
| `format_alpaca` | Parses instruction/output format (Alpaca, Dolly) |
| `format_orca` | Parses system_prompt/question/response format (OpenOrca) |
| `format_truthfulqa` | Parses question/best_answer format (TruthfulQA) |
| `_flush` | Writes a batch of conversations to a numbered part file |

```bash
python collect/sharegpt.py --preset ultrachat --limit 50000
python collect/sharegpt.py --dataset teknium/OpenHermes-2.5 --format sharegpt --out raw/hermes
```

---

### Step 2 — Data Preparation (`collect/prepare.py`)

Takes all the `.txt` files from `raw/` and converts them into binary token files for fast training. This only needs to run once — the output `.bin` files are what `train.py` actually reads.

| Function | What it does |
|---|---|
| `find_text_files` | Recursively finds all `.txt` files under `raw/` |
| `train_tokenizer` | Samples 200MB of text and trains the BPE vocabulary on it |
| `_worker_init` | Initializes each CPU worker process with a copy of the tokenizer |
| `_encode_file` | Worker function that tokenizes one file and returns raw bytes |
| `encode_corpus` | Runs all workers in parallel, streams output to `train.bin` and `val.bin` |

```bash
python collect/prepare.py --raw raw/ --out data/ --vocab 32000
```

**Output:**
- `data/train.bin` — ~90% of all tokens, used for training
- `data/val.bin` — ~10% of tokens, used to measure overfitting
- `data/tokenizer.json` — saved BPE merge rules

---

### Step 3 — Tokenizer (`data.py`)

A from-scratch BPE tokenizer with no external dependencies. Converts raw text into integer token IDs and back.

| Function | What it does |
|---|---|
| `BPETokenizer.train` | Learns merge rules by repeatedly merging the most frequent byte pair |
| `BPETokenizer.encode` | Converts a string to a list of token IDs using learned merges |
| `BPETokenizer._encode_bytes` | Applies merges to a single chunk of bytes |
| `BPETokenizer.decode` | Converts token IDs back to a UTF-8 string |
| `BPETokenizer.state` | Returns a JSON-serializable dict of all merges and vocab entries |
| `BPETokenizer.from_state` | Reconstructs a tokenizer from a saved state dict |

---

### Step 4 — Model (`model.py`)

The transformer architecture. Four classes that stack on top of each other to form `BabyGPT`.

| Class | What it does |
|---|---|
| `CausalSelfAttention` | Multi-head attention — each token attends to all previous tokens. Uses Flash Attention internally for speed |
| `MLP` | Feed-forward layer: Linear → GELU → Linear with 4× expansion |
| `Block` | One full transformer layer: LayerNorm → Attention → residual, LayerNorm → MLP → residual |
| `BabyGPT` | The full model — stacks N blocks, token + position embeddings, produces logits. Has a `generate` method for inference |

---

### Step 5 — Pretraining (`train.py`)

Trains BabyGPT to predict the next token across all text. This is where the model learns language, facts, and reasoning patterns.

| Function | What it does |
|---|---|
| `get_lr` | Cosine learning rate schedule with linear warmup |
| `get_batch_text` | Samples a random window from in-memory text data (small datasets) |
| `get_batch_bin` | Samples a random window from a memory-mapped `.bin` file (large datasets) |
| `estimate_val_loss` | Runs 50 batches on the validation set and returns average loss |
| `_save` | Saves model weights, optimizer state, and metadata to `.pt`. Deletes the previous step checkpoint to save disk space |
| `main` | The full training loop: loads data, builds model, runs steps, logs loss, saves checkpoints |

```bash
python train.py \
  --train-bin data/train.bin \
  --val-bin   data/val.bin \
  --tokenizer data/tokenizer.json \
  --n-embd 1920 --n-layer 24 --n-head 16 \
  --steps 100000 --bf16 --grad-checkpoint
```

---

### Step 6 — Fine-tuning (`finetune.py`)

Takes the pretrained checkpoint and teaches it to be a chatbot. The key difference from pretraining: loss is only computed on assistant turns, not user turns. This teaches the model to respond rather than just continue text.

| Function | What it does |
|---|---|
| `parse_turns` | Extracts `(role, content)` pairs from one conversation block |
| `load_conversations` | Reads all `.txt` files in `raw/` and returns parsed conversations |
| `tokenize_conversation` | Tokenizes a conversation and builds a per-token loss mask (1 = assistant, 0 = user) |
| `SFTDataset` | Packs all conversations end-to-end, samples random windows for training |
| `SFTDataset.get_batch` | Returns `(x, y, mask)` — mask zeroes out user tokens so they don't contribute to loss |
| `sft_loss` | Forward pass that only computes cross-entropy on assistant tokens |
| `estimate_val_loss` | Same as `train.py` but uses `sft_loss` |
| `get_lr` | Same cosine schedule but lower peak LR than pretraining (default `1e-4`) |
| `_save` | Saves step checkpoint and also writes `sft_checkpoint_best.pt` whenever val_loss hits a new low |
| `main` | The fine-tuning loop |

```bash
python finetune.py \
  --checkpoint checkpoint.pt \
  --data raw/openhermes raw/dolly \
  --steps 10000 --lr 1e-4 --bf16 --grad-checkpoint
```
