# Tensomite: handoff

What the next session, or the next person, needs in order to carry on without the history of the last one. `PLAN.md`, `DESIGN.md` and `DEVELOPMENT.md` hold the project itself. This document holds how we work, where things stand and what has been learned.

**Keeping it current.** Rewrite it rather than appending to it: at the end of each PR, and whenever a session's context passes about 70%. Anything that belongs in one of the other three documents goes there instead.

*Last rewritten: 23 Oct 2026, with M6's last PR open.*

## Where things stand

- **On `main`:** milestones 1 to 5, and M6 parts 1 to 3: the lazy tensor core, the CPU op set and `SimDiscrete`, the BLAS seam, GPT-2 inference verified against PyTorch, autograd with SGD, AdamW and a Shakespeare trainer, and a `Metal` device with every `F32` op on the GPU. CI runs on Linux and macOS (the macOS runner has a GPU), and Decisions A–S are recorded in `PLAN.md`.
- **Open PR:** M6 part 4 on `m6-metal-split-k`: split-K matmul, Decision S, `bench/matvec.cr`, and the M6 exit spec on GPT-2 small. It completes M6.
- **M6 exit, all met:** every op matches the oracle (`gpu_spec.cr`); GPT-2 small on Metal matches PyTorch, decodes greedily like it, and gives bit-identical prefill and decode (CI's GPU and the M5 Pro); Metal decodes at 64 tokens/s against BLAS's 152 and prefills at 1,149 against 906; determinism costs 0–4% for matrix-vector products and 8% for prefill (`DESIGN.md`, "Performance baselines").
- **Next:** M7, kernel macros and memory planning. Measurements point at its targets: a token's matmuls take 3 ms of Metal's 16, so the rest is dispatches and one-thread-per-row reductions. Decision O requires Metal to beat BLAS there.
- **Deferred:** Decision D, CPU parallelism. Profiling in M5 shows that training is bound by elementwise passes, so D now has a measured case; M7's fusion comes first.

## Working agreements

Agreement           |Practice                                                                                                                                                                                                                                         
--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Branches            |One per PR, named for its purpose (`m5-autograd`, `docs-handoff`), cut from the latest `main`.                                                                                                                                                   
Pull requests       |Claude opens them without waiting to be asked, then watches them: CI, reviews and merge, with a safety-net check-in.                                                                                                                             
Stacked PRs         |When asked. The later PR's base is the earlier branch; GitHub retargets it after the earlier one merges.                                                                                                                                         
Merging             |The maintainer merges, by squash or by merge commit. After a squash, merge `main` into a stacked branch and resolve conflicts with the branch's side; never rewrite history or force-push (the session blocks it anyway).                        
Milestones          |Each ends with its exit criterion passing, with the evidence (numbers, logs) in the PR description.                                                                                                                                              
Documents           |Updated in the same PR as the code (see "Workflow" in `DEVELOPMENT.md`). Tables and diagrams follow "Documents" in `DEVELOPMENT.md`; check them with `scripts/format_tables.py` and `scripts/check_diagrams.sh`.                                 
Decisions           |Labelled with letters and recorded in `PLAN.md`. The next is **T**; S allowed Metal's matmul to differ from the CPU in the last bits. P (how to get GPT-2 small's fixtures) was settled by allowing Hugging Face; Q made this document committed.
Understanding checks|Short questions labelled Q-A, Q-B and so on, after explaining something new. The next is **Q-S**.                                                                                                                                                
Claims              |Measured before stated. Several assumptions were wrong when checked: decode time was dominated by the KV cache, not host kernels; GPT-2's file was unaligned. Say "I think" when something is unverified.                                        

## Follow-ups found on the way

Item                                   |Why                                                                           |Where it fits                    
---------------------------------------|------------------------------------------------------------------------------|---------------------------------
Preallocated KV cache, written in place|`Cat` still copies the whole cache every token, though in runs now            |Any time; small                  
Buffer reuse                           |Training allocates hundreds of MB per step; the GC's unmapping doubled a step |M7 (memory planning)             
Fewer passes over memory               |A dense pass costs 2.4 ns per element in training, against 0.45 ns when cached|M7 (fusion)                      
Parallel host kernels                  |Training is bound by elementwise passes                                       |Decision D                       
Reuse scalar leaves                    |Each `x * 0.5` uploads a new rank-0 tensor                                    |M7 (kernel macros remove most)   
`Int32` element counts                 |Caps one tensor at 2 GiB                                                      |Before models with larger tensors

## Environment

### Claude Code cloud container

Nothing persists between containers except what is pushed. Setting one up from scratch:

1. **Crystal 1.22.0:** download the official tarball from GitHub releases into `/opt`, then link it: `ln -sf /opt/crystal-1.22.0-1/bin/* /usr/local/bin/`. Only 1.22.0 existed on GitHub at the time.
2. **OpenBLAS:** `apt-get install -y libopenblas-dev`.
3. **Shards and ameba:** see "Setup" in `DEVELOPMENT.md`.
4. **Python, for fixtures only:** a venv with Python 3.13 (PyTorch has no wheels for 3.15), then `pip install torch transformers tokenizers safetensors`. Keep the venv in the scratchpad, not the repo.
5. **GPT-2 small:** `scripts/fetch_gpt2.sh models/gpt2`.
6. **Diagram checks:** `PUPPETEER_EXECUTABLE_PATH=/opt/pw-browsers/chromium PUPPETEER_SKIP_DOWNLOAD=1 scripts/check_diagrams.sh`.

**Network.** Outbound HTTPS goes through a policy proxy. A 403 means the host is not allowed, so don't retry; ask for it to be added. Hugging Face needs all three entries: `huggingface.co`, `*.huggingface.co` and `*.hf.co`. A wildcard does not cover the bare domain, and large files redirect to a `*.hf.co` CDN. PyPI and GitHub are allowed by default.

**Training data:** `scripts/fetch_shakespeare.sh` (from GitHub raw, allowed by default).

**Long runs.** Start them with `nohup … &` and a log in the scratchpad, then wait with a background `until grep -q … log; do sleep 30; done`; a foreground `sleep` is blocked. Set `GC_UNMAP_THRESHOLD=0` for training. The container is reclaimed when the session goes idle, which kills background jobs: a run of several hours survives only while the session is active, so run those on the maintainer's machine.

**Shell gotchas.** `rm` on a path built from a shell variable (`rm -f "$D"/*`) is blocked; use a literal path or `"${D:?}"`. Force-pushes are blocked. Crystal cannot `require` an absolute path (see "Crystal gotchas" in `DEVELOPMENT.md`), so temporary scripts that load tensomite go inside the repo and are deleted afterwards.

### Maintainer's machine

A MacBook Pro with an M5 Pro, using Accelerate. Its benchmark results are in "Performance baselines" in `DESIGN.md`. A Python venv there must also use 3.13 or older until PyTorch supports newer releases.

## Concepts covered

The maintainer prefers to understand rather than consume, through analogies, examples and Mermaid diagrams. These have been explained and checked, so build on them rather than re-explaining:

Concept                      |Takeaway                                                                                                            |Check   
-----------------------------|--------------------------------------------------------------------------------------------------------------------|--------
Lazy graphs                  |Ops build nodes; `realize` runs them, so passes can see the whole graph. Shape errors still come early.             |Q-A     
Discrete and unified memory  |Design for transfers; unified memory makes them free, not the reverse.                                              |Q-B     
Shader compilation           |Source to SPIR-V can be at build time; SPIR-V to machine code is always the driver's.                               |none    
Fusion                       |One kernel instead of many, saving memory traffic and dispatches. Compile-time macros fuse what is written together.|Q-E, Q-F
Matmul and fusion            |Matmul reuses inputs, so it is tiled; only its epilogue and prologue fuse.                                          |Q-E     
Online softmax               |A running maximum with rescaling by `exp(m − m')` lets softmax work tile by tile, as in FlashAttention.             |Q-G     
Associativity and determinism|Float addition order changes the bits. Levels 1, 2 and 3 of Decision K.                                             |Q-H, Q-I
Prefix caching               |Why level 2 matters: a cache hit must give the same bits as recomputing.                                            |Q-I     
Auto-vectorisation           |`Slice` vectorises and `Array` does not; float sums do not reorder; integer sums do.                                |none    
Loop-carried dependencies    |Vectorise across independent outputs; a running total can't be split without reordering.                            |Q-O     
Quantised dot products       |Q4 × Q8 in integers vectorises and is exact within a block.                                                         |none    
Where BLAS's speed comes from|Bookkeeping, loop order, register and cache blocking, threads, matrix units: factors that multiply.                 |Q-N     
Decode is memory-bound       |Each token streams all the weights; the GPU's edge shrinks on unified memory.                                       |none    
Reverse-mode autograd        |`grad` is a graph transform: chain rules build a backward graph of ordinary ops; views' adjoints sum (`Unview`).    |not yet 
Split-K and determinism      |Fixed-size chunks of `k`, combined in chunk order, keep bits independent of row count and zero-padding; costs ~0–8%.|Q-R     

Questions asked but never answered: Q-D (why a SPIR-V emitter in Crystal needs no compiler at run time), Q-J (why `sum(axis)` permutes the axis to the end), Q-L (why `Cat` writes through a slice view), Q-M (why `max` starts from the first element), Q-P (why broadcasting needs no gradient rule of its own), Q-Q (the gradient of `x` expanded from `[3]` to `[2, 3]`), Q-R (why the split must depend only on `k` for prefill to equal decode). Q-K was answered, after a correction: a stride of 0 means repeated reads of one position, not contiguous rows.
