# BabyGPT
My own LLM built entirely from scratch (In Effort to spread knowledge on LLM development)

BabyGPT is A small but real language model built entirely from scratch using Python and PyTorch, no shortcuts at all .

With BabyGPT you feed it any `.txt` file.  It learns the statistical patterns in that text and responds like a traditional chatbot.
This initially started as an NLP, detecting text patterns and predicting the next words that finish the sentence.

    Ex. User: "May the force"

        BabyGPT: "be with you"

This model has a **Decoder-only Transformer** architecture, commoonly used in most popular LLM's.
The tokenizer was built from scratch initially starting as a **Character-Level tokenizer** (which eventually turned out better suited for NLP's) and evolved into a **Byte-Pair Encoding (BPE) tokenizer**, commonly used in GPT models.
I'll go over the pros, cons, and differences, and defintions of everything later on. But for now try running it for yourself. I've included my BabyGPT 1.1B param model, small enough to run on your laptop ;)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect training data

Download conversation datasets from HuggingFace (or anywhere you choose) into `raw/`:

Note * I made this mistake very early on while researching and building this model. Pretraining is purely for the model to learn words (tokens), Pretraining data is reserved for teaching the model its
knowledge. Facts, Grammar, writing style etc. 

- For a well rounded model I suggest searching hugging face and finding some already put together raw .txt data. These are a few of the common popular websites I decided to use.

- The pretraining phase is essentially what gives the model its set of weights in every matrice layer.

- Finetuning, is strictly meant for tuning your model to recieve questions, statements etc. and respond like a traditional chatbot. Try to use chat style raw .txt files when finetuning, in the format of
    User: "question_here?"
    Assistant: "response_here"

- I've added some flags to help format data that is not in this current format. I'll explains the flags later on !

P.S: again make sure all data is a raw .txt file !!

P.S x2: Web Scraping is not the best approach, again you want a model that is a knowleageable and accurate as possible, so if you web scrape your data make sure to clean it of any innaccurate noise. Remeber, keep ethics and morality in mind when creating your LLM pwease kitten !!!

```bash
Pre-Training Data
python collect/wikipedia.py --out raw/wikipedia --limit 100000
python collect/github.py --languages python javascript --out raw/github --limit 500000
python collect/reddit.py --out raw/reddit --limit 200000
python collect/stackoverflow.py --out raw/stackoverflow --limit 500000
python collect/arxiv.py --abstracts-only --out raw/arxiv --limit 100000
python collect/common_crawl.py --out raw/common_crawl --limit 1000000

Finetuning Data
python collect/sharegpt.py --preset ultrachat --limit 50000 --out raw/chat_ultra
python collect/sharegpt.py --dataset teknium/OpenHermes-2.5 --format sharegpt --out raw/openhermes --limit 50000
python collect/sharegpt.py --dataset databricks/dolly-15k --format alpaca --out raw/dolly
python collect/sharegpt.py --dataset Open-Orca/OpenOrca --format orca --out raw/openorca --limit 50000
python collect/sharegpt.py --dataset truthful_qa --config generation --format truthfulqa --out raw/truthfulqa
python collect/sharegpt.py --dataset OpenAssistant/oasst1 --format messages --out raw/openassistant
python collect/sharegpt.py --dataset garage-bAInd/Open-Platypus --format alpaca --out raw/platypus
python collect/sharegpt.py --dataset WizardLM/WizardLM_evol_instruct_V2_196k --format sharegpt --out raw/wizardlm --limit 50000
python collect/sharegpt.py --dataset HuggingFaceH4/CodeAlpaca_20K --format alpaca --out raw/codealpaca
python collect/sharegpt.py --preset vicuna --limit 50000 --out raw/sharegpt
python collect/gutenberg.py --out raw/gutenberg
```

### 3. Tokenize the corpus

Convert all `.txt` files into binary token files for fast training:

```bash
python collect/prepare.py --raw raw/ --out data/ --vocab 32000
```

This will create `data/train.bin`, `data/val.bin`, and `data/tokenizer.json`.

### 4. Pretrain the model

```bash
python train.py \
  --train-bin data/train.bin \
  --val-bin   data/val.bin \
  --tokenizer data/tokenizer.json \
  --n-embd 1920 --n-layer 24 --n-head 16 \
  --steps 100000 --bf16 --grad-checkpoint \
  --out checkpoint.pt
```

Adjust `--n-embd`, `--n-layer`, `--n-head` to control model size. Add `--adam8bit` if you run out of VRAM.

### 5. Fine-tune into a chatbot

```bash
python finetune.py \
  --checkpoint checkpoint.pt \
  --data raw/ultrachat raw/openhermes raw/dolly \
  --steps 10000 --lr 1e-4 \
  --bf16 --grad-checkpoint \
  --out sft_checkpoint.pt
```

### 6. Chat with the model

**Terminal:**
```bash
python chat.py --checkpoint sft_checkpoint.pt
```

**Web app** (from the `llm-web/` directory):
```bash
cd ../llm-web
pip install -r requirements.txt
CHECKPOINT=../llm/sft_checkpoint.pt uvicorn app:app --host 0.0.0.0 --port 8000
```
Then open `http://localhost:8000`.

## Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Primary language (and the best language (argue with a wall) ) |
| **PyTorch** | [Tensors](https://docs.pytorch.org/docs/stable/tensors.html) , [automatic gradients](https://docs.pytorch.org/docs/stable/autograd.html) , and GPU acceleration |
| **tqdm** | Gives you some pwetty progress bars while training |

---

## Here Is the High-Level Flow of it all my dudes

```
data.txt  ──►  data.py  ──►  train.py  ──►  checkpoint.pt ──► finetune.py ──► stf_checkpoint.pt
                                                                                       │
                                                                                llm-web/app.py  ──►  localhost:8000
```

1. `data.py` trains a BPE tokenizer that maps text to integer token ids
2. `train.py` encodes le data and teaches le model to predict le next token (gives the model it's weights via the tokenizer thingy above this thingy)
3. Trained weights + tokenizer are saved to `checkpoint.pt` (there is checkpointing at every 1000 steps so you can resume if something happens)
4. `chat.py` loads the checkpoint and lets you type prompts in your terminal (I highly reccommend using my llm-web version instead, its so pwetty)

---

## Now into the good stuff, the real DOCUMENTATION >:)

Originally this was **not** a chatbot in the traditional sense, instead it was a **next-token predictor**.  it's only job was to predict what token comes next given an input. But then I got curious >:) !!

I learned that ChatGPT, Claude, Gemini etcc. all **started** as base language models trained pretty much like this one, a beefed up NLP. While I was in school I remember learning a lot about NLP and more specifically cleaning and tokenizing text. 
I read this lovely book textbook [Natural Language Processing by Jacob Eisenstein](eisenstein_nlp_textbook (1)) and really didn't see a vision into how this was used, which is why this started as a next token predictor instead of a true chatbot style LLM.

But then I learned about the capabilites of PyTorch, the different types of tokenizers, capabilities of tensors allowing GPU acceleration, backpropogation, gradient descent, and most importantly model weights and instruction tuning. This is where finetune.py was added !

I'll dive into each file and their functionality further, but I do want to give you some vocabulary and their importance as a pre step to understanding whats going on in the next step.

Lets start with the nitty gritty stuff

    - Neural Network: A function that takes numbers in, transforms them through layers of math, and produces numbers out. The math in each layer is just matrix multiplication followed by a non-linear activation function.
    
    - Weights: The actual learned numbers inside the model. Every linear layer, every attention projection, every embedding is a matrix of floating point numbers (Tensors via PyTorch). Everything the model knows is stored as these numbers. Training is just the process of finding the right values for all of them.
    
    - Loss: A single number that measures how wrong the model's prediction was. Lower = better. The model outputs a probability distribution over the entire vocabulary — loss measures how much probability it assigned to the correct next token. If the correct token got 90% probability, loss is low. If it got 1% probability, loss is high.
        - There's Val Loss: loss on held-out data the model has never seen. This is the best measure of model quality.
        - And theres also Train loss: loss on the data the model is actively learning from
        
    - Backpropagation: The algorithm that figures out how much each weight contributed to the error
    
    - Cross-Entropy: Measures the difference between the model's predicted probability distribution and the true distribution (which is 100% on the correct next token, 0% everywhere else) and penalizes the model heavily when it assigns low probability to the correct answer.
    
    - Embedding: Converting a token ID (an integer) into a dense vector of floats
    
    - Logits: Raw unnormalized scores the model outputs for every token in the vocabulary before softmax. The highest logit is the model's best guess for next token.
    
    - Softmax: converts logits into probabilities that sum to 1.0. Ex. [2.1, 0.3, -1.4] turns into [0.76, 0.13, 0.11]

    - CausalSelfAttention: The function that lets each token look back at previous tokens and decide which ones are relevant to its current meaning.

These all work off of eachother, I tried to order them to the best of my ability to give you an understanding of how they work together.

Next I want to go through some of the vocab I had to experienced through trial and error when I actually started training this thing, somethings to avoid and keep an eye on::

    - Learning Rate: Controls how big each weight update is after every backward pass. It's a very very small number
    
    - Overfitting: In this context, this is when val loss rises while train loss keeps falling The model is starting memorizing the data instead of prediccting. This leads to  your model hallucinating
    
    - Underfitting; When your model hasn't trained enough data, and it doesn't know what to do resulting in high val lass and train loss
    
    - Divergence: When your model isn't learning anything, it probably means your learning rate is too high. You can tell when your val loss over time keeps climbing instead of decreasing.

    - Parameters: Essentially another word for weights, it's every single learnable number in the model. The parameter count is the total of all of them added together.

The general rule's are:

| Thingy's | Rule|
| --- | --- |
| **Learning Rate** | If too high, the loss will bounce back and forht and you'll miss the lowest loss, if too low it'll take 10x longer, If just right it'll have a nice descent down to a nice val loss |
| **Overfitting** | Keep an eye out on your val loss and train loss, val loss should be around 1.2 for a good model, keeping in mind you need to finetune |
| **Parameters** | The larger Param model, the more raw data you'll need to pretrain on, or else your model will indeed hallucinate (I did this because I was lazy, and wasted alot of money training it on RunPod lmaoo) |

Here is a good guide for where your val loss should be on each stage of training:

Pretraining

| Val Loss | Quality |
|---|---|
| > 3.0 | Still learning, pretty early in training |
| 2.0 – 3.0 | Meh |
| 1.5 – 2.0 | Its Aight |
| 1.2 – 1.5 | Pretty Solid I would say |
| < 1.2 | Very Solid, but be careful not to overfit mah boy |

Fine-tuning

| Val Loss | Quality |
|---|---|
| > 1.4 | Still learning|
| 1.2 – 1.4 | Usable|
| 1.0 – 1.2 | Good|
| 0.8 – 1.0 | Pretty Solid I would say |
| < 0.8 | Very Solid, but be careful not to overfit mah boy |

---

## File-by-File Breakdown

---

### `data.py`: BPE Tokenizer

This file converts raw text into numbers the model can work with, and back again (Encoding, Decoding).

Since neural networks don't read text and instead read integers, we need to tokenzie the words and encode an integer to them, which is where BPE (Byte-Pair Encoding) comes in. BPE is the algorithm that decides which sequences of characters (words lol) get their own integer id.

#### Why BPE instead of continuing with character-level?

Character-level tokenization assigns one id/integer per character (`h`, `e`, `l`, `l`, `o` = 5 tokens).  BPE merges frequent id pairs until common words and subwords get single ids, so now (`hello` = 1 token).

This matters for two reasons:
1. **Compression**: Fewer tokens means the context window (the amount of input and output info an llm can remember at once) covers more actual text.
2. **Richer signal per step**: The model predicts `hello` as one unit instead of predicting each letter individually, which is a much harder and less meaningful task.

#### How BPE training works (step by step)

```
Every character becomes its raw byte value (0–255):

 "hello" → [104, 101, 108, 108, 111]
                                 (one integer per byte, 0-255)

Step 1: We would then count all adjacent pairs, and grab the most frequent pair
        [104, 101]
        [101, 108]
        [108, 108]  <-- Most frequent pair
        [108, 111]
        

Step 2: We then merge the most frequent pair and assign the pair with a new token ID (starting at 256) and replace every occurence of that pair in the corpus (raw txt data) with the new Token ID
    now "ll" in "hello"
    [108,108] or "ll" is assigned the token ID 256
    now the "hello" = [104, 101, 256, 111]

Step 3: We now update the vocabulary, its byte representation is the concatenation of the two tokens that were merged
    ex. the byte representation of "l" is simply "l"
        so "l" + "l" = "ll"
        now in your vocabulary -> vocab[256] = b'll'

Step 4: We repeat the process by counting pairs again, finding the most frequent pair, assigning a new id, and updating the vocab (Keep in mind, this is done on the entire corpus, not just one word at a time)
        [104,101] -> he
        [101,256] -> ell
        [256,111] -> llo
        lets pick "ell"
        now vocab[257] = b'ell'

        [104,257] -> hell (hahaaa)
        [257,111] -> ello
        lets pick "ello"
        now vocab[258] = b'ello'

        you should be getting it by now lol
        eventually you will now have an entire word 'hello' assigned to 1 token id
```

#### `BPETokenizer` class

| Method | What it does |
|---|---|
| `train(text, vocab_size)` | Counts pair frequencies, merges the most common pair into a new token, repeats until `vocab_size` is reached |
| `encode(text)` | Converts a string to bytes, then applies the learned merges (highest-priority first) until no more apply |
| `decode(ids)` | Maps each id back to its byte sequence, concatenates, decodes as UTF-8 (allows me to reference the id back to the original text) |
| `state()` | Returns a JSON-serialisable dict of all merges and vocab entries (this is saved inside the checkpoint.pt file) |
| `from_state(data)` | Reconstructs a tokenizer from a dict produced by `state()` |

Read this table and the functions available in the BPETokenizer class, keeping in mind the steps I just explained above this table.

**Vocabulary layout:**

| Id range | What it holds |
|---|---|
| 0 – 255 | Raw bytes (always present; these are the "base" tokens) |
| 256+ | Merged tokens learned during training |

Because every possible byte is already in the vocabulary, the encoder can **never** encounter an unknown token.  Any valid UTF-8 text can be encoded.
---

#### `CausalSelfAttention`: The "Reading" Mechanism

The heart of the Transformer in model.py. It Lets the model learn which previous tokens are most relevant when predicting the next one.

**Q, K, V — Queries, Keys, Values:**

Think of it like a search engine for each token position:
- **Query (Q):** "What am I looking for?"
- **Key (K):**   "What do I offer?"
- **Value (V):** "What information do I contain?"

The dot product `Q × Kᵀ` scores how relevant each past position is.  Dividing by `√head_dim` keeps the values numerically stable (prevents softmax from becoming saturated). The `softmax` essentially turns scores into probabilities (an "attention map"), then finally `att × V` collects the weighted sum of information from all past positions.

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

**What is cross-entropy loss?**
The model outputs a probability distribution over the vocabulary so, cross-entropy measures how different that distribution is from the "correct" distribution (probability 1.0 on the actual next token, 0.0 everywhere else).  Minimizing it forces the model to assign high probability to the correct next token.

**What is AdamW?**
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

## How to Improve the Quality

Quality is a product of three things:

1. **Data** — More and more varied text = better. The model will sound like whatever it is trained on.
2. **Tokenizer** — Larger `--bpe_vocab_size` means richer tokens and better compression (keep in mind, it does take longer to train).
3. **Model size** — Increase `--n_embd`, `--n_layer`, `--n_head` for more capacity, more parameters = more patterns the model can store (more parameters means you need a larger corpus, dont replicate my mistakes).
4. **Training time** — Keep an eye on the loss value, the lower loss the better predictions, be sure not to overfit though.

```bash
# Bigger vocabulary, bigger model, longer training
python train.py --data data.txt --epochs 20 --bpe_vocab_size 2000 --n_embd 256 --n_layer 6 --n_head 8
```

---

## My Next Steps

| Step | Status | What it adds |
|---|---|---|
| **RLHF** | Pending | Reinforcement learning, basically takes in human feedback and helps the model understand preferred human responses |
| **Learn Vision Modeling** | Working on it | I want to build a robotic arm that can speak to me and help me troubleshoot hardware projects by looking at them :) |

---

### Here are some function explanations

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

### `collect/prepare.py`

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

### `data.py`

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

### `model.py`

The transformer architecture. Four classes that stack on top of each other to form `BabyGPT`.

| Class | What it does |
|---|---|
| `CausalSelfAttention` | Multi-head attention — each token attends to all previous tokens. Uses Flash Attention internally for speed |
| `MLP` | Feed-forward layer: Linear → GELU → Linear with 4× expansion |
| `Block` | One full transformer layer: LayerNorm → Attention → residual, LayerNorm → MLP → residual |
| `BabyGPT` | The full model — stacks N blocks, token + position embeddings, produces logits. Has a `generate` method for inference |

---

### `train.py`

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

### `finetune.py`

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
