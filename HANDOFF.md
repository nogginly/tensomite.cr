# Tensomite: handoff

What the next session, or the next person, needs in order to carry on without the history of the last one. `PLAN.md`, `DESIGN.md` and `DEVELOPMENT.md` hold the project itself. This document holds how we work, where things stand and what has been learned.

**Keeping it current.** Rewrite it rather than appending to it: at the end of each PR, and whenever a session's context passes about 70%. Anything that belongs in one of the other three documents goes there instead.

*Last rewritten: 23 Oct 2026, after M4 merged.*

## Where things stand

- **On `main`:** milestones 1 to 4. That covers the lazy tensor core, the CPU op set and `SimDiscrete`, the BLAS seam, and GPT-2 inference verified against PyTorch on GPT-2 small. CI runs on Linux and macOS, and Decisions A–O are recorded in `PLAN.md`.
- **Open PRs:** none, apart from the one adding this document.
- **Next:** M5, autograd and training, as described in `PLAN.md`. Realised nodes currently drop their sources (see "Tensor and Node" in `DESIGN.md`); autograd has to revisit that first.
- **Deferred:** Decision D, CPU parallelism, waits on profiling.

## Working agreements

Agreement           |Practice                                                                                                                                                                                                                 
--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Branches            |One per PR, named for its purpose (`m5-autograd`, `docs-handoff`), cut from the latest `main`.                                                                                                                           
Pull requests       |Claude opens them without waiting to be asked, then watches them: CI, reviews and merge, with a safety-net check-in.                                                                                                     
Stacked PRs         |When asked. The later PR's base is the earlier branch; GitHub retargets it after the earlier one merges.                                                                                                                 
Merging             |The maintainer merges, by squash or by merge commit. After a squash, merge `main` into a stacked branch and resolve conflicts with the branch's side; never rewrite history or force-push (the session blocks it anyway).
Milestones          |Each ends with its exit criterion passing, with the evidence (numbers, logs) in the PR description.                                                                                                                      
Documents           |Updated in the same PR as the code (see "Workflow" in `DEVELOPMENT.md`). Tables and diagrams follow "Documents" in `DEVELOPMENT.md`; check them with `scripts/format_tables.py` and `scripts/check_diagrams.sh`.         
Decisions           |Labelled with letters and recorded in `PLAN.md`. The next is **Decision R**. P (how to get GPT-2 small's fixtures) was settled by allowing Hugging Face; Q made this document committed.                                 
Understanding checks|Short questions labelled Q-A, Q-B and so on, after explaining something new. The next is **Q-P**.                                                                                                                        
Claims              |Measured before stated. Several assumptions were wrong when checked: decode time was dominated by the KV cache, not host kernels; GPT-2's file was unaligned. Say "I think" when something is unverified.                

## Follow-ups found on the way

Item                                               |Why                                                               |Where it fits                     
---------------------------------------------------|------------------------------------------------------------------|----------------------------------
Preallocated KV cache, written in place            |`Cat` is 48% of GPT-2 decode time on `BLAS`                       |Any time; small                   
Contiguous fast paths for elementwise ops and `Cat`|Every element goes through `View#offset_of`                       |M11, or earlier if profiles demand
Iterative graph walk in `Scheduler`                |The recursive walk could exhaust the stack on deep training graphs|M5                                
Reuse scalar leaves                                |Each `x * 0.5` uploads a new rank-0 tensor                        |M7 (kernel macros remove most)    
`Int32` element counts                             |Caps one tensor at 2 GiB                                          |Before models with larger tensors 

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

Questions asked but never answered: Q-D (why a SPIR-V emitter in Crystal needs no compiler at run time), Q-J (why `sum(axis)` permutes the axis to the end), Q-K (what a stride of 0 means for reads), Q-L (why `Cat` writes through a slice view), Q-M (why `max` starts from the first element).
