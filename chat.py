from __future__ import annotations

import argparse

import torch

from data import BPETokenizer
from model import BabyGPT



def main() -> None:
    p = argparse.ArgumentParser(description='Chat with a trained TinyGPT model')
    p.add_argument('--checkpoint',     type=str,   default='checkpoint.pt')
    p.add_argument('--max-new-tokens', type=int,   default=300,
                   help='Maximum tokens to generate per turn')
    p.add_argument('--temperature',    type=float, default=0.7,
                   help='Sampling temperature: lower = focused, higher = creative')
    p.add_argument('--top-k',          type=int,   default=50,
                   help='Top-k filtering (0 = disabled)')
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    meta = ckpt['meta']

    tokenizer = BPETokenizer.from_state(meta['tokenizer'])

    model = BabyGPT(
        vocab_size = meta['vocab_size'],
        block_size = meta['block_size'],
        n_layer    = meta['n_layer'],
        n_head     = meta['n_head'],
        n_embd     = meta['n_embd'],
    )
    model.load_state_dict(ckpt['model'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    is_sft = meta.get('sft', False)
    top_k  = args.top_k if args.top_k > 0 else None

    if is_sft:
        print('SFT chatbot loaded.  Type a message and press Enter.')
        print('Type "reset" to clear conversation history.')
        print('Ctrl-C or empty line to exit.\n')
    else:
        print('Pretrained model loaded.  Type a prompt to continue text.')
        print('Ctrl-C or empty line to exit.\n')

    _STOP_PREFIXES = ('A:', 'Q:', 'User:', 'Assistant:', '---')
    block_size     = meta['block_size']
    history        = []   # list of (user, assistant) string pairs

    while True:
        try:
            user_input = input('You> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            break

        if user_input.lower() == 'reset':
            history = []
            print('Conversation history cleared.\n')
            continue

        if is_sft:
            prompt = f'User: {user_input}\n\nAssistant: '

            while len(tokenizer.encode(prompt)) > block_size - args.max_new_tokens and history:
                history.pop(0)
                turns = ''
                for u, a in history:
                    turns += f'User: {u}\n\nAssistant: {a}\n\n'
                prompt = turns + f'User: {user_input}\n\nAssistant: '
        else:
            prompt = user_input

        ids = tokenizer.encode(prompt)
        if not ids:
            print('Could not encode input.\n')
            continue

        ctx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(
            ctx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
        )

        full_text = tokenizer.decode(out[0].tolist())

        if is_sft:
            response = full_text[len(prompt):]
            clean_lines = []
            for line in response.split('\n'):
                if line.strip().startswith(_STOP_PREFIXES):
                    break
                clean_lines.append(line)
            response = '\n'.join(clean_lines).strip()

            history.append((user_input, response))

            print(f'Bot> {response}\n')
        else:
            print(f'Bot> {full_text}\n')


if __name__ == '__main__':
    main()
