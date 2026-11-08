"""Generates the reference fixtures that tensomite's model and tokeniser specs compare against.

    python scripts/make_fixtures.py tiny                # spec/fixtures/tiny_gpt2/
    python scripts/make_fixtures.py gpt2 GPT2_DIR       # spec/fixtures/gpt2/
    python scripts/make_fixtures.py tokenizer           # spec/fixtures/tokenizer/
    python scripts/make_fixtures.py smollm2 SMOLLM2_DIR # spec/fixtures/smollm2/

`tokenizer` needs only `tokenizers`: it trains a small byte-level BPE with
SmolLM2's pre-tokeniser (digits split, then GPT-2's pattern) and ChatML's
special tokens, saves it as tokenizer.json, and records the ids it gives.
`smollm2` records the ids SMOLLM2_DIR's tokenizer.json gives; the file itself
is not committed.

`tiny` needs only PyPI packages (torch, transformers, tokenizers, safetensors):
it trains a small byte-level BPE vocabulary and builds a randomly initialised
2-layer GPT-2, then records what the reference implementations compute.

`gpt2` needs GPT2_DIR to hold GPT-2 small's model.safetensors, vocab.json and
merges.txt (see scripts/fetch_gpt2.sh). The weights are not committed; only the
expected outputs are.

Both write expected.json with:
  corpus, corpus_ids    tokeniser inputs and their ids
  prompt, prompt_ids    a prompt and its ids
  logits                logits of every prompt position (tiny) or of the first
                        and last positions (gpt2), as nested lists
  greedy                greedy continuation of the prompt
"""

import json
import pathlib
import sys

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "spec" / "fixtures"

# Text that exercises the pre-tokeniser: contractions, numbers, runs of spaces,
# tabs and newlines, Unicode whitespace, accents, combining marks, CJK, emoji with
# joiners and right-to-left script.
CORPUS = [
    "Hello, world!",
    "It's a test: they'll, we've, I'm, you'd, she's, we're.",
    "Numbers 12345 and 3.14159, plus 1,000,000.",
    "  leading spaces,\ttabs\tand\nnew\n\nlines  ",
    "Unicode spaces: a b c　d.",
    "Café naïve résumé, café with a combining accent.",
    "日本語のテキストと中文文本。",
    "Emoji 🙂 and a family 👩‍👩‍👧 and flags 🇬🇧.",
    "مرحبا بالعالم",
    "Symbols: <>{}[]~`@#$%^&*()_+-=|\\/ — “quotes” – dashes…",
    "",
]

PROMPT = "The quick brown fox jumps over the lazy dog. It's"

# ChatML's special tokens, and text that mixes them with digits and ordinary text.
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
CHAT = [
    "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nWhat is 12 + 345?<|im_end|>\n<|im_start|>assistant\n",
    "a<|im_end|>b<|im_start|><|im_end|> 2024-11-08 ٣٤ ³",
    "<|im_start|",
]


def tokenizer_ids(vocab: pathlib.Path, merges: pathlib.Path):
    """Encodes the corpus with transformers' Python GPT-2 tokeniser, the original algorithm."""
    from transformers import GPT2Tokenizer

    tok = GPT2Tokenizer(str(vocab), str(merges))
    return tok, [tok.encode(text) for text in CORPUS]


def write_expected(out: pathlib.Path, tok, corpus_ids, model, logits_rows, greedy_count):
    import torch

    prompt_ids = tok.encode(PROMPT)
    with torch.no_grad():
        logits = model(torch.tensor([prompt_ids])).logits[0]
        greedy = model.generate(
            torch.tensor([prompt_ids]),
            max_new_tokens=greedy_count,
            do_sample=False,
            pad_token_id=0,
            attention_mask=torch.ones(1, len(prompt_ids), dtype=torch.long),
        )[0][len(prompt_ids):]
    rows = logits_rows(len(prompt_ids))
    expected = {
        "corpus": CORPUS,
        "corpus_ids": corpus_ids,
        "prompt": PROMPT,
        "prompt_ids": prompt_ids,
        "logit_rows": rows,
        "logits": [logits[row].tolist() for row in rows],
        "greedy": greedy.tolist(),
    }
    (out / "expected.json").write_text(json.dumps(expected, ensure_ascii=False))


def tiny():
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    out = FIXTURES / "tiny_gpt2"
    out.mkdir(parents=True, exist_ok=True)

    # 1. Train a small byte-level BPE vocabulary on the corpus and prompt.
    bpe = Tokenizer(models.BPE())
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    bpe.train_from_iterator(CORPUS + [PROMPT] * 3, trainer)
    bpe.model.save(str(out))

    # 2. Encode the corpus with both reference tokenisers; they must agree.
    tok, corpus_ids = tokenizer_ids(out / "vocab.json", out / "merges.txt")
    fast_ids = [bpe.encode(text).ids for text in CORPUS]
    assert corpus_ids == fast_ids, "the Python and Rust reference tokenisers disagree"

    # 3. Build a random GPT-2 with weights large enough that logits are far apart.
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=bpe.get_vocab_size(), n_positions=64, n_embd=32, n_layer=2, n_head=4,
        initializer_range=0.3, activation_function="gelu_new",
    )
    model = GPT2LMHeadModel(config).eval()
    model.save_pretrained(str(out), safe_serialization=True)
    for extra in ("config.json", "generation_config.json"):
        (out / extra).unlink(missing_ok=True)

    write_expected(out, tok, corpus_ids, model, lambda count: list(range(count)), 20)


def gpt2(directory: str):
    from transformers import GPT2LMHeadModel

    source = pathlib.Path(directory)
    out = FIXTURES / "gpt2"
    out.mkdir(parents=True, exist_ok=True)
    tok, corpus_ids = tokenizer_ids(source / "vocab.json", source / "merges.txt")
    model = GPT2LMHeadModel.from_pretrained(str(source)).eval()
    write_expected(out, tok, corpus_ids, model, lambda count: [0, count - 1], 64)


def write_token_ids(out: pathlib.Path, tok):
    """Records the ids *tok* gives the corpus and the chat texts, with and
    without special tokens, and the text each decodes back to."""
    texts = CORPUS + CHAT
    expected = {
        "texts": texts,
        "ids": [tok.encode(text, add_special_tokens=False).ids for text in texts],
        "decoded": [tok.decode(tok.encode(text).ids, skip_special_tokens=False) for text in texts],
    }
    (out / "expected.json").write_text(json.dumps(expected, ensure_ascii=False))


def tokenizer():
    out = FIXTURES / "tokenizer"
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400, special_tokens=SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(CORPUS + CHAT + [PROMPT] * 3, trainer)
    tok.save(str(out / "tokenizer.json"))
    write_token_ids(out, tok)


def smollm2(directory: str):
    out = FIXTURES / "smollm2"
    out.mkdir(parents=True, exist_ok=True)
    write_token_ids(out, Tokenizer.from_file(str(pathlib.Path(directory) / "tokenizer.json")))


if __name__ == "__main__":
    if sys.argv[1:2] == ["tiny"]:
        tiny()
    elif sys.argv[1:2] == ["gpt2"] and len(sys.argv) == 3:
        gpt2(sys.argv[2])
    elif sys.argv[1:2] == ["tokenizer"]:
        tokenizer()
    elif sys.argv[1:2] == ["smollm2"] and len(sys.argv) == 3:
        smollm2(sys.argv[2])
    else:
        sys.exit(__doc__)
