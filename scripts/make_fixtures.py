"""Generates the reference fixtures that tensomite's model and tokeniser specs compare against.

    python scripts/make_fixtures.py tiny                # spec/fixtures/tiny_gpt2/
    python scripts/make_fixtures.py gpt2 GPT2_DIR       # spec/fixtures/gpt2/
    python scripts/make_fixtures.py tokenizer           # spec/fixtures/tokenizer/
    python scripts/make_fixtures.py smollm2 SMOLLM2_DIR # spec/fixtures/smollm2/
    python scripts/make_fixtures.py tiny_llama          # spec/fixtures/tiny_llama/
    python scripts/make_fixtures.py tiny_gguf           # spec/fixtures/tiny_llama/*.gguf
    python scripts/make_fixtures.py quant               # spec/fixtures/quant/

`tokenizer` needs only `tokenizers`: it trains a small byte-level BPE with
SmolLM2's pre-tokeniser (digits split, then GPT-2's pattern) and ChatML's
special tokens, saves it as tokenizer.json, and records the ids it gives.
`smollm2` records the ids SMOLLM2_DIR's tokenizer.json gives, and the logits
and greedy continuation of a ChatML prompt; the model is not committed.
`quant` records what gguf-py's reference quantisers, which match ggml's
bit for bit, make of fixed rows: random values, ties, constant and zero
blocks, as hex bytes and dequantised values for Q8_0, Q4_0 and Q4_1.
`tiny_gguf` writes `tiny_llama` and `tokenizer` as GGUF files, at F32, F16,
Q8_0 and Q4_0 (every matrix quantised, the norms kept at F32),
as llama.cpp's converter does, with each head's query and key rows reordered.
`tiny_llama` builds a randomly initialised 2-layer Llama, with grouped queries
and an untied head, over `tokenizer`'s vocabulary, and records its logits.

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


CHAT_PROMPT = "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"


def write_llama_logits(out: pathlib.Path, tok, model, prompt: str, logits_rows, greedy_count: int, eos: int):
    """Records *model*'s logits for *prompt* at *logits_rows* and its greedy continuation."""
    import torch

    prompt_ids = tok.encode(prompt).ids
    with torch.no_grad():
        logits = model(torch.tensor([prompt_ids])).logits[0]
        greedy = model.generate(
            torch.tensor([prompt_ids]), max_new_tokens=greedy_count, min_new_tokens=greedy_count,
            do_sample=False, pad_token_id=eos, eos_token_id=None,
            attention_mask=torch.ones(1, len(prompt_ids), dtype=torch.long),
        )[0][len(prompt_ids):]
    rows = logits_rows(len(prompt_ids))
    expected = {
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "logit_rows": rows,
        "logits": [logits[row].tolist() for row in rows],
        "greedy": greedy.tolist(),
    }
    (out / "logits.json").write_text(json.dumps(expected, ensure_ascii=False))


def smollm2(directory: str):
    import torch
    from transformers import LlamaForCausalLM

    source = pathlib.Path(directory)
    out = FIXTURES / "smollm2"
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(source / "tokenizer.json"))
    write_token_ids(out, tok)
    model = LlamaForCausalLM.from_pretrained(str(source), torch_dtype=torch.float32).eval()
    write_llama_logits(out, tok, model, CHAT_PROMPT, lambda count: [0, count - 1], 32, 2)


def tiny_llama():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    out = FIXTURES / "tiny_llama"
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(FIXTURES / "tokenizer" / "tokenizer.json"))
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=tok.get_vocab_size(), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64, rope_theta=10000.0,
        rms_norm_eps=1e-5, tie_word_embeddings=False, initializer_range=0.3,
        bos_token_id=1, eos_token_id=2, pad_token_id=2,
    )
    model = LlamaForCausalLM(config).eval()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm.weight"):
                param.uniform_(0.5, 1.5)
    model.save_pretrained(str(out), safe_serialization=True)
    (out / "generation_config.json").unlink(missing_ok=True)
    write_llama_logits(out, tok, model, CHAT_PROMPT, lambda count: list(range(count)), 20, 2)


def gguf_permute(weights, heads: int):
    """Reorders each head's rows so that rotary pairs are adjacent, as
    llama.cpp's convert_hf_to_gguf.py does to q_proj and k_proj."""
    return weights.reshape(heads, 2, weights.shape[0] // heads // 2, *weights.shape[1:]).swapaxes(1, 2).reshape(weights.shape)


def tiny_gguf():
    import gguf
    import numpy as np
    from safetensors.numpy import load_file

    source = FIXTURES / "tiny_llama"
    config = json.loads((source / "config.json").read_text())
    weights = load_file(str(source / "model.safetensors"))
    tok = json.loads((FIXTURES / "tokenizer" / "tokenizer.json").read_text())
    vocab = sorted(tok["model"]["vocab"].items(), key=lambda item: item[1])
    added = {token["id"] for token in tok["added_tokens"]}
    names = {
        "input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm",
        "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
        "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
        "mlp.down_proj": "ffn_down",
    }
    for suffix, kind in (("", np.float32), ("-f16", np.float16), ("-q8_0", "Q8_0"), ("-q4_0", "Q4_0")):
        writer = gguf.GGUFWriter(str(source / f"model{suffix}.gguf"), "llama")
        writer.add_name("tiny llama")
        writer.add_block_count(config["num_hidden_layers"])
        writer.add_context_length(config["max_position_embeddings"])
        writer.add_embedding_length(config["hidden_size"])
        writer.add_feed_forward_length(config["intermediate_size"])
        writer.add_head_count(config["num_attention_heads"])
        writer.add_head_count_kv(config["num_key_value_heads"])
        writer.add_rope_freq_base(config["rope_parameters"]["rope_theta"])
        writer.add_layer_norm_rms_eps(config["rms_norm_eps"])
        writer.add_vocab_size(config["vocab_size"])
        writer.add_rope_dimension_count(config["head_dim"])
        writer.add_tokenizer_model("gpt2")
        writer.add_tokenizer_pre("smollm")
        writer.add_token_list([token for token, _ in vocab])
        writer.add_token_types([3 if id in added else 1 for _, id in vocab])
        writer.add_token_merges([merge if isinstance(merge, str) else " ".join(merge) for merge in tok["model"]["merges"]])
        writer.add_bos_token_id(1)
        writer.add_eos_token_id(2)
        for name, tensor in weights.items():
            if name.startswith("model.layers."):
                _, _, layer, rest = name.split(".", 3)
                part, last = rest.rsplit(".", 1)
                key = f"blk.{layer}.{names[part]}.{last}"
                if part == "self_attn.q_proj":
                    tensor = gguf_permute(tensor, config["num_attention_heads"])
                elif part == "self_attn.k_proj":
                    tensor = gguf_permute(tensor, config["num_key_value_heads"])
            else:
                key = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight",
                       "lm_head.weight": "output.weight"}[name]
            if tensor.ndim == 1:
                writer.add_tensor(key, tensor.astype(np.float32))
            elif isinstance(kind, str):
                qtype = getattr(gguf.GGMLQuantizationType, kind)
                writer.add_tensor(key, gguf.quants.quantize(tensor.astype(np.float32), qtype), raw_dtype=qtype)
            else:
                writer.add_tensor(key, tensor.astype(kind))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()


def quant():
    import gguf
    import numpy as np

    out = FIXTURES / "quant"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = [
        rng.normal(0, 1, 64),
        rng.uniform(0.1, 0.5, 32),
        np.linspace(-1, 1, 32),
        np.full(32, 0.25),
        np.zeros(32),
        np.array([(i - 16) / 2 for i in range(32)]),
        rng.normal(0, 1e-3, 32),
    ]
    rows = [row.astype(np.float32) for row in rows]
    expected = {"rows": [row.tolist() for row in rows]}
    for kind in ("Q8_0", "Q4_0", "Q4_1"):
        qtype = getattr(gguf.GGMLQuantizationType, kind)
        blocks = [gguf.quants.quantize(row.reshape(1, -1), qtype) for row in rows]
        expected[kind] = {
            "bytes": [block.tobytes().hex() for block in blocks],
            "values": [gguf.quants.dequantize(block, qtype).reshape(-1).tolist() for block in blocks],
        }
    table = rng.normal(0, 1, (3, 64)).astype(np.float32)
    writer = gguf.GGUFWriter(str(out / "blocks.gguf"), "llama")
    expected["table"] = table.tolist()
    for kind in ("Q8_0", "Q4_0", "Q4_1"):
        qtype = getattr(gguf.GGMLQuantizationType, kind)
        blocks = gguf.quants.quantize(table, qtype)
        writer.add_tensor(kind.lower(), blocks, raw_dtype=qtype)
        expected[kind]["table"] = gguf.quants.dequantize(blocks, qtype).tolist()
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    (out / "expected.json").write_text(json.dumps(expected))


if __name__ == "__main__":
    if sys.argv[1:2] == ["tiny"]:
        tiny()
    elif sys.argv[1:2] == ["gpt2"] and len(sys.argv) == 3:
        gpt2(sys.argv[2])
    elif sys.argv[1:2] == ["tokenizer"]:
        tokenizer()
    elif sys.argv[1:2] == ["smollm2"] and len(sys.argv) == 3:
        smollm2(sys.argv[2])
    elif sys.argv[1:2] == ["tiny_llama"]:
        tiny_llama()
    elif sys.argv[1:2] == ["tiny_gguf"]:
        tiny_gguf()
    elif sys.argv[1:2] == ["quant"]:
        quant()
    else:
        sys.exit(__doc__)
