"""Converts a tiktoken tokeniser, such as autoresearch's tokenizer.pkl, to a
Hugging Face tokenizer.json that tensomite's `BPE.from_json` reads.

    python scripts/tiktoken_to_json.py tokenizer.pkl tokenizer.json

The pickle holds a `tiktoken.Encoding`, as autoresearch's prepare.py saves it.
Each token's bytes become GPT-2's byte-level characters; its merges are
recovered in rank order, each token split into the two lower-ranked tokens
tiktoken's own merging would join; the split pattern becomes a `Split`
pre-tokeniser before a `ByteLevel` with its own pattern off; and the special
tokens become added tokens. Needs `tiktoken` (pip install tiktoken).
"""

import json
import pickle
import sys


def byte_to_char():
    """GPT-2's map from each byte to a printable character."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    chars = printable[:]
    n = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            chars.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(printable, chars)}


def split_token(ranks, token: bytes):
    """Returns the two tokens tiktoken joins last to make *token*: merging its
    bytes by rank, using only tokens ranked below it."""
    limit = ranks[token]
    parts = [bytes([b]) for b in token]
    while len(parts) > 2:
        best, best_rank = None, None
        for i in range(len(parts) - 1):
            rank = ranks.get(parts[i] + parts[i + 1])
            if rank is not None and rank < limit and (best_rank is None or rank < best_rank):
                best, best_rank = i, rank
        if best is None:
            raise ValueError(f"token {token!r} cannot be built from lower-ranked tokens")
        parts[best:best + 2] = [parts[best] + parts[best + 1]]
    return parts


def convert(encoding) -> dict:
    ranks = encoding._mergeable_ranks
    chars = byte_to_char()
    text = lambda token: "".join(chars[b] for b in token)  # noqa: E731
    vocab = {text(token): rank for token, rank in ranks.items()}
    merges = []
    for token, _ in sorted(ranks.items(), key=lambda item: item[1]):
        if len(token) > 1:
            left, right = split_token(ranks, token)
            merges.append(f"{text(left)} {text(right)}")
    added = [{"id": i, "content": s, "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": True}
             for s, i in sorted(encoding._special_tokens.items(), key=lambda item: item[1])]
    return {
        "version": "1.0",
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": encoding._pat_str}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
        ]},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "vocab": vocab,
                  "merges": merges},
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    with open(sys.argv[1], "rb") as file:
        encoding = pickle.load(file)
    with open(sys.argv[2], "w", encoding="utf-8") as file:
        json.dump(convert(encoding), file, ensure_ascii=False)
