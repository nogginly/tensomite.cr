---
name: finetune-a-model
description: Use when asked to fine-tune a pretrained GPT-2 (small, medium, large or XL) on a text file with tensomite's apps/finetune, to resume or stop early a run, or to serve the result with apps/server. Not for Llama-family models, and not for training a network from scratch (use train-a-model).
---

# Fine-tune GPT-2 with tensomite

`apps/finetune` fine-tunes a pretrained GPT-2 on a text file. A run lives in its own directory, checkpoints as it goes, resumes where it stopped, validates, stops early, and exports a model directory that `apps/server` serves. In a dependent project the shard is in `lib/tensomite/`; paths below are relative to the tensomite checkout, so prefix them with `lib/tensomite/` there. Run the commands from the project root.

## Steps

1. **Build the tools.** Both apps are `shards` targets of the tensomite shard. From a dependent project's root, compile them into `bin/`:

   ```sh
   crystal build --release -o bin/finetune lib/tensomite/apps/finetune/finetune.cr
   crystal build --release -o bin/server lib/tensomite/apps/server/server.cr
   ```

   Inside a tensomite checkout, `shards build finetune server --release` builds them into `bin/`. Add `-Dcuda` for an NVIDIA GPU. `bin/finetune --help` lists every option.

2. **Get GPT-2 weights.** The script downloads `model.safetensors`, `vocab.json`, `merges.txt` and `config.json`, skipping files already there:

   ```sh
   lib/tensomite/scripts/fetch_gpt2.sh models/gpt2     # GPT-2 small, 124M parameters
   lib/tensomite/scripts/fetch_gpt2.sh medium          # into models/gpt2-medium; also large, xl
   ```

3. **Prepare the text.** Any plain text file works. The run tokenises it once and keeps the last tenth as validation text. For a first try, `lib/tensomite/scripts/fetch_shakespeare.sh data/shakespeare` writes Tiny Shakespeare to `data/shakespeare/input.txt`, the default for `--data`.

4. **Start a run.** `--out` names the run directory and is required:

   ```sh
   bin/finetune --out runs/mine --model models/gpt2 --data my-text.txt --device blas --steps 2000 --patience 5
   ```

   `--device` is `cpu`, `blas` (the default), `metal` on Apple Silicon, or `cuda` in a `-Dcuda` build. Every 10 steps it prints the loss, learning rate, milliseconds a step and memory; every `--eval-every` steps it prints the validation loss and a sample continuing `--prompt`. The same lines are appended to `runs/mine/log.txt`.

5. **Main options**, as `bin/finetune --help` and `apps/finetune/README.md` give them:

   Option                          |Default         |Meaning                                                                                
   --------------------------------|----------------|---------------------------------------------------------------------------------------
   `--model`                       |`models/gpt2`   |the pretrained GPT-2; its heads come from its `config.json`, or `--heads`              
   `--data`                        |Tiny Shakespeare|the text to fine-tune on                                                               
   `--batch`, `--context`          |8, 256          |sequences per step, and tokens per sequence                                            
   `--steps`                       |1000            |training steps                                                                         
   `--lr`, `--min-lr`, `--warmup`  |1e-4, 1e-5, 50  |a linear warm-up to the peak, then a cosine decay to the minimum                       
   `--weight-decay`                |0.01            |AdamW's pull of the weights towards zero                                               
   `--dropout`                     |0.1             |the share of activations zeroed at each training step; 0 for none                      
   `--eval-every`, `--eval-batches`|100, 8          |steps between validations, and batches in each                                         
   `--patience`                    |0               |validations without a new best before stopping; 0 never stops early                    
   `--prompt`                      |a newline       |what each validation's sample continues                                                
   `--checkpoint-every`            |100             |steps between checkpoints                                                              
   `--seed`                        |1337            |the random seed                                                                        

6. **Checkpoints and resuming.** The run directory holds `settings.json`, `tokens.bin` and `split.json`, `checkpoint/` (weights, AdamW's `m` and `v`, the step), `best/` (the weights at the best validation loss), `model/` (the weights at the end) and `log.txt`. After an interruption, run the command again with `--out` and, if needed, `--device`:

   ```sh
   bin/finetune --out runs/mine --device blas
   ```

   A resumed run reads its settings from `settings.json`; only `--device` and `--checkpoint-every` apply. On one device it takes the same steps, to the bit, as a run that never stopped.

7. **Stop early.** A small text is soon memorised, and validation loss then rises. Set `--patience N` to stop after N validations without a new best, and serve `best/` rather than `model/`. On Tiny Shakespeare with GPT-2 small, the best validation loss came at step 200 without dropout and at step 500 with 0.1.

8. **Serve the result.** `best/` and `model/` are model directories that `apps/server` reads:

   ```sh
   bin/server --model runs/mine/best                   # http://127.0.0.1:8080, OpenAI-style /v1/completions
   bin/server generate --model runs/mine/best "Once upon a time" --max-tokens 40 --temperature 0.8
   ```

   From Crystal, `GPT2.load("runs/mine/best/model.safetensors", heads: 12)` and `BPE.from_files("runs/mine/best/vocab.json", "runs/mine/best/merges.txt")` load it; `heads` is 16, 20 and 25 for medium, large and XL (`GUIDE.md`, "Pretrained models: GPT-2").

## Time and memory

- GPT-2 small at batch 8 × 256 trains at about 5,000 tokens a second on an NVIDIA GB10 and 2,700 on an Apple M5 Pro, so a night is roughly 150M and 75M tokens.
- Fine-tuning keeps a gradient and AdamW's `m` and `v` beside every weight: GPT-2 small needs about 2 GB for them in F32, before activations.
- GPT-2 large at batch 4 × 512 takes 2.7 s a step on the GB10, with about 34 GB of device memory. Lower `--batch` or `--context` when memory runs out.

## Limits

- GPT-2 family only, small to XL. Llama-family models such as SmolLM2 load for inference (`Llama.load`, `apps/server`), but `apps/finetune` cannot fine-tune them yet.
- F32 only, and no LoRA: every weight is trained, with its gradient and AdamW state in memory.
- A run uses one device, and resumes bit for bit only on that device.
- The text is tokenised with GPT-2's own vocabulary; there is no tokeniser training.

## Where to read more

- `apps/finetune/README.md` (`lib/tensomite/apps/finetune/README.md`): a run in detail, exact resuming, fine-tuning on code.
- `apps/server/README.md` (`lib/tensomite/apps/server/README.md`): serving, the HTTP API, chat.
- `GUIDE.md` (`lib/tensomite/GUIDE.md`), "Pretrained models: GPT-2": loading, generating and fine-tuning from Crystal code.
- The `train-a-model` skill: defining and training a network of your own.
