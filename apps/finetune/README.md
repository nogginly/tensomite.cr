# finetune

Fine-tunes a pretrained GPT-2 on a text file, built on tensomite. A run lives in its own directory, checkpoints as it goes, resumes where it stopped, and exports a model `apps/server` serves.

```sh
scripts/fetch_gpt2.sh                                    # GPT-2 small; `medium`, `large` or `xl` for larger
ops br finetune                                          # builds bin/release/finetune
bin/release/finetune --out runs/mine --data my-text.txt --device metal --steps 2000 --patience 5
bin/release/finetune --out runs/mine --device metal      # after an interruption: resumes
bin/release/server --model runs/mine/best                # serves the best weights
```

`ops run-finetune ARGS` builds and runs it in one step. `--device` picks `cpu`, `blas` (the default), `metal` on Apple Silicon, or `cuda` in a build with `-Dcuda`.

## A run

```mermaid
flowchart LR
    text["text file"] -->|"once"| tokens["tokens.bin: 16-bit ids, memory-mapped"]
    tokens --> train["first nine tenths: training windows"]
    tokens --> val["last tenth: fixed validation windows"]
    train --> step["step: batch from (seed, step), forward, loss, backward, AdamW"]
    step -->|"every --eval-every"| check["validation loss and a sample"]
    check -->|"new best"| best["best/"]
    check -->|"no new best in --patience checks"| stop["stop"]
    step -->|"every --checkpoint-every"| ckpt["checkpoint/: weights, AdamW's m and v, step"]
    ckpt -.->|"resume"| step
    step -->|"last step"| model["model/"]
    style ckpt stroke:#1971c2,stroke-width:3px
    style best stroke:#2b8a3e,stroke-width:3px
```

The run directory holds:

File or directory         |What it is                                                                                                         
--------------------------|-------------------------------------------------------------------------------------------------------------------
`settings.json`           |every setting, written by the first run; a resumed run reads it, and only `--device` and `--checkpoint-every` apply
`tokens.bin`, `split.json`|the text, tokenised once, and where its validation tenth begins                                                    
`checkpoint/`             |the weights, AdamW's state and the progress: step, best validation loss, checks since the best                     
`best/`                   |the weights at the best validation loss, with the tokeniser and config: a model directory                          
`model/`                  |the weights at the end, the same way                                                                               
`log.txt`                 |the lines printed, appended across resumes                                                                         

## Resuming exactly

A checkpoint holds the weights, AdamW's `m` and `v`, and the step. The rest of what decides the next step is already fixed. The batch of step *n* comes from a generator seeded with the run's seed and *n*, and so does its dropout, so no generator state needs saving. The learning rate is a function of the step and the settings. The text is the saved `tokens.bin`, not the text file, which may since have changed. So on one device, a run resumed from a checkpoint takes the same steps, to the bit, as one that never stopped; the specs check this. Each checkpoint is written to `checkpoint.new/` and then swapped in, so an interruption leaves one whole checkpoint.

## Settings

Option                          |Default         |Meaning                                                                                
--------------------------------|----------------|---------------------------------------------------------------------------------------
`--model`                       |`models/gpt2`   |the pretrained GPT-2; its heads come from its `config.json`                            
`--data`                        |Tiny Shakespeare|the text to fine-tune on                                                               
`--batch`, `--context`          |8, 256          |sequences per step, and tokens per sequence                                            
`--steps`                       |1000            |training steps                                                                         
`--lr`, `--min-lr`, `--warmup`  |1e-4, 1e-5, 50  |a linear warm-up to the peak, then a cosine decay to the minimum                       
`--weight-decay`                |0.01            |AdamW's pull of the weights towards zero                                               
`--dropout`                     |0.1             |the share of activations zeroed at each training step, as GPT-2 was trained; 0 for none
`--eval-every`, `--eval-batches`|100, 8          |steps between validations, and batches in each                                         
`--patience`                    |0               |validations without a new best before stopping; 0 never stops early                    
`--prompt`                      |a newline       |what each validation's sample continues                                                

**How long a run takes.** GPT-2 small at batch 8 × 256 trains at about 5,000 tokens/s on a GB10 and 2,700 on an M5 Pro, so a night is roughly 150M and 75M tokens. A small text is seen many times over: Tiny Shakespeare's 300K tokens would pass about 1,200 times in a night on the GB10. Set `--patience` so the run stops when validation stops improving, rather than memorising the text. On Tiny Shakespeare with GPT-2 small, the best validation loss came at step 200 without dropout and at step 500 with 0.1, which also kept validation loss nearly level for hundreds of steps after its best.

## Specs

```sh
crystal spec apps/finetune/spec
```

They train the tiny GPT-2 of `spec/fixtures/tiny_gpt2` on the CPU on a generated text: the corpus file, the learning-rate schedule, batches drawn from the seed and step alone, a resumed run equal bit for bit to an uninterrupted one, the loss falling, and the best weights kept and exported with early stopping.
