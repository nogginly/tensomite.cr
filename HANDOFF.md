# Tensomite: handoff

What the next session, or the next person, needs in order to carry on without the history of the last one. `PLAN.md`, `DESIGN.md` and `DEVELOPMENT.md` hold the project itself. This document holds how we work, where things stand and what has been learned.

**Keeping it current.** Rewrite it rather than appending to it: at the end of each PR, and whenever a session's context passes about 70%. Anything that belongs in one of the other three documents goes there instead.

*Last rewritten: 31 Oct 2026, with shared scalar constants open.*

## Where things stand

- **On `main`:** milestones 1 to 7 and 7b. That covers the lazy tensor core, the CPU op set and `SimDiscrete`, the BLAS seam, GPT-2 inference verified against PyTorch, autograd with SGD, AdamW and a Shakespeare trainer, and the `Metal` device. M7 added parallel reductions on Metal, a thread-safe buffer pool with batched commits, kernel macros, fused row ops, vectorised host loops with tensomite's own `exp`, `log` and `tanh`, graph diagrams with scopes, and a tiled matmul with split-K's bits. CI runs on Linux and macOS (the macOS runner has a GPU), and Decisions A–AR are recorded in `PLAN.md`. M7b put every model on `Tensomite::Module`s, built with a `Builder`, with `save` and `load` by name in safetensors files.
- **M7's exit, met** (evidence under M7 in `PLAN.md`):
  - 316 dispatches per decoded GPT-2 token, from 710;
  - Metal decodes at 186 tokens/s against BLAS's 166, and prefills 64 tokens at 3,684 against 2,419;
  - macro kernels match their oracles and their gradients pass finite differences;
  - Shakespeare reached 1.573 nats/char in 20 minutes in the Linux container.
- **M7b's exit, met** (evidence under 7b in `PLAN.md`): the tutorial, Shakespeare and GPT-2 models on modules with the same bits as the hand-written ones, on every device; the `Builder` chosen as the DSL (Decisions AN and AR).
- **Open PR:** scalar constants shared by device, dtype and value, so a training step uploads each once (`m8-scalar-cache`, Decision BK): 7 uploads a step on the CUDA stub, from 205, with the same losses; not yet measured on a GPU. Merged: `MatMulMode::TiledTwoPass` for products with few blocks of outputs and a long `k` (Decision BJ), which took a training step of the larger model to 22.4 ms on the M5 Pro, from 59.1 before Decision BI, and 17.2 on the GB10, from 20.5; Metal's `Unview` loading 32 elements ahead; Decision BI, Metal's `Unview`, parallel `ScatterAdd` zeroing and one wait per step; profiling for both GPUs (Decision BH); the trainer on CUDA (Decision BG).
- **Now: M8, a CUDA backend** (Decisions AS–AV in `PLAN.md`; Vulkan moved to M12). Development is on the maintainer's DGX Spark, the exit on a cloud T4. Nothing in the container runs CUDA kernels: the maintainer runs them on the Spark, as with Metal on the Mac. The container and CI run the `CUDA` device and its kernels on `scripts/cuda_stub`, which compiles `kernels.cu` as host C++. Next: cross-entropy's gradient without `ScatterAdd` (Decision BK's second part), then the rest of part 5, the exit runs on the Spark and a T4.
- **A long-term goal:** inspecting running models (activations, attention, gradients) by the names of their parts; see "Inspectable" in `PLAN.md` and "Towards inspection" in `DESIGN.md`. Scopes are its first piece, and Decision AA must keep inspected nodes alive.
- **Deferred:** Decision D, CPU threads (not needed for M7's exits). Decision AA, releasing buffers as soon as their last reader has run.

## Working agreements

Agreement           |Practice                                                                                                                                                                                                                                                             
--------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Branches            |One per PR, named for its purpose (`m5-autograd`, `docs-handoff`), cut from the latest `main`.                                                                                                                                                                       
Pull requests       |Claude opens them without waiting to be asked, then watches them: CI, reviews and merge, with a safety-net check-in.                                                                                                                                                 
Stacked PRs         |When asked. The later PR's base is the earlier branch; GitHub retargets it after the earlier one merges.                                                                                                                                                             
Merging             |The maintainer merges, by squash or by merge commit. After a squash, merge `main` into a stacked branch and resolve conflicts with the branch's side; never rewrite history or force-push (the session blocks it anyway).                                            
Milestones          |Each ends with its exit criterion passing, with the evidence (numbers, logs) in the PR description.                                                                                                                                                                  
Documents           |Updated in the same PR as the code (see "Workflow" in `DEVELOPMENT.md`), `GUIDE.md` included whenever the public API changes. Tables and diagrams follow "Documents" in `DEVELOPMENT.md`; check them with `scripts/format_tables.py` and `scripts/check_diagrams.sh`.
Decisions           |Labelled with letters and recorded in `PLAN.md`. The next is **BL**; T to AG shape M7, AH to AR shape M7b, and AS to BK shape M8. P (how to get GPT-2 small's fixtures) was settled by allowing Hugging Face; Q made this document committed.                        
Understanding checks|Short questions labelled Q-A, Q-B and so on, after explaining something new. The next is **Q-AU**; Q-T to Q-AT are recent.                                                                                                                                           
Claims              |Measured before stated. Several assumptions were wrong when checked: decode time was dominated by the KV cache, not host kernels; GPT-2's file was unaligned. Say "I think" when something is unverified.                                                            

## Follow-ups found on the way

Item                        |Why                                                                                                                                                                                                                                                   |Where it fits                                         
----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------
A declarative model macro   |`ln1(x)` rather than `@ln1.call(x)` in `forward`; the macro tried in M7b (`parts ... do ln1 = layernorm(width) end`) made ameba report each part as a useless assignment, so a new one needs a syntax ameba accepts without suppressions (Decision AR)|When a model's `forward` grows long                   
Narrow Metal matmuls        |Products with 768 output columns run at about 950 GFLOP/s, Accelerate's half; splitting `k` across threadgroups in split-K's order means storing each chunk's sum (Decision AG)                                                                       |When prefill matters again                            
Hardware matrix instructions|`simdgroup_matrix` (and the M5's GPU matrix units, I think) could beat the tiled kernel, but their order of addition is unspecified                                                                                                                   |An opt-in mode with relaxed determinism, like `Atomic`
Planned buffer release      |Lower peak memory, and buffer lifetimes independent of the collector (Decision AA)                                                                                                                                                                    |After M7                                              
Fewer passes over memory    |A host pass costs 0.7 ns per element in training; the remaining fusions (attention's scale and mask, AdamW, bias adds) would save some                                                                                                                |Optional (Decision Z)                                 
Parallel host kernels       |Training is bound by elementwise passes                                                                                                                                                                                                               |Decision D                                            
Reuse scalar leaves         |Each `x * 0.5` uploads a new rank-0 tensor                                                                                                                                                                                                            |Any time; kernel macros removed most                  
`Int32` element counts      |Caps one tensor at 2 GiB                                                                                                                                                                                                                              |Before models with larger tensors                     

## Environment

### Claude Code cloud container

Nothing persists between containers except what is pushed. Setting one up from scratch:

1. **Crystal 1.22.0:** download the official tarball from GitHub releases into `/opt`, then link it: `ln -sf /opt/crystal-1.22.0-1/bin/* /usr/local/bin/`. Only 1.22.0 existed on GitHub at the time.
2. **OpenBLAS:** `apt-get install -y libopenblas-dev`.
3. **Shards and ameba:** see "Setup" in `DEVELOPMENT.md`.
4. **Python, for fixtures only:** a venv with Python 3.13 (PyTorch has no wheels for 3.15), then `pip install torch transformers tokenizers safetensors`. Keep the venv in the scratchpad, not the repo.
5. **GPT-2 small:** `scripts/fetch_gpt2.sh models/gpt2`.
6. **Diagram checks:** `PUPPETEER_EXECUTABLE_PATH=/opt/pw-browsers/chromium PUPPETEER_SKIP_DOWNLOAD=1 scripts/check_diagrams.sh`.
7. **Metal kernels without a Mac:** `scripts/emulate_metal/run.sh` compiles the matmul kernels as C++ and checks the tiled ones against split-K. The Metal compiler itself runs only on macOS: in CI's macOS job, or on the maintainer's machine. CI runs only on pull requests, so open a draft PR to get a kernel compiled early.

**Network.** Outbound HTTPS goes through a policy proxy. A 403 means the host is not allowed, so don't retry; ask for it to be added. Hugging Face needs all three entries: `huggingface.co`, `*.huggingface.co` and `*.hf.co`. A wildcard does not cover the bare domain, and large files redirect to a `*.hf.co` CDN. PyPI and GitHub are allowed by default.

**Training data:** `scripts/fetch_shakespeare.sh` (from GitHub raw, allowed by default).

**High `sys` time** in training is OpenBLAS's threads busy-waiting, not the collector; `OPENBLAS_NUM_THREADS=1` removes it.

**Specs run in random order** (`ops test_specs`); a run ends with `Randomized with seed: N`, and `crystal spec --order N` replays it.

**Long runs.** Start them with `nohup … &` and a log in the scratchpad, then wait with a background `until grep -q … log; do sleep 30; done`; a foreground `sleep` is blocked. The container is reclaimed when the session goes idle, which kills background jobs: a run of several hours survives only while the session is active, so run those on the maintainer's machine.

**Shell gotchas.** `rm` on a path built from a shell variable (`rm -f "$D"/*`) is blocked; use a literal path or `"${D:?}"`. Force-pushes are blocked. Crystal cannot `require` an absolute path (see "Crystal gotchas" in `DEVELOPMENT.md`), so temporary scripts that load tensomite go inside the repo and are deleted afterwards.

### Maintainer's machine

A MacBook Pro with an M5 Pro, using Accelerate. Its benchmark results are in "Performance baselines" in `DESIGN.md`. A Python venv there must also use 3.13 or older until PyTorch supports newer releases.

An NVIDIA DGX Spark for CUDA: a GB10 on ARM64, its GPU sharing 128 GB with the CPU, running headless Ubuntu Server with NVIDIA's compute-only driver 595 (CUDA 13.2) and Crystal 1.22.1. Nothing graphical is to be installed on it, which is why Vulkan waits (Decision AT). NVRTC is Ubuntu's `cuda-nvrtc-13-1` and `cuda-nvrtc-dev-13-1`, in `/usr/local/cuda-13.1/targets/sbsa-linux/lib`, added to the loader's path with a file in `/etc/ld.so.conf.d`; the linker still needs `LIBRARY_PATH` set to that directory. The maintainer runs CUDA specs and examples there, as with Metal on the Mac.

## Concepts covered

The maintainer prefers to understand rather than consume, through analogies, examples and Mermaid diagrams. These have been explained and checked, so build on them rather than re-explaining:

Concept                      |Takeaway                                                                                                                                                               |Check          
-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------
Lazy graphs                  |Ops build nodes; `realize` runs them, so passes can see the whole graph. Shape errors still come early.                                                                |Q-A            
Discrete and unified memory  |Design for transfers; unified memory makes them free, not the reverse.                                                                                                 |Q-B            
Shader compilation           |Source to SPIR-V can be at build time; SPIR-V to machine code is always the driver's. For CUDA, NVRTC turns source into PTX and the driver turns PTX into machine code.|none           
Fusion                       |One kernel instead of many, saving memory traffic and dispatches. Compile-time macros fuse what is written together.                                                   |Q-E, Q-F       
Matmul and fusion            |Matmul reuses inputs, so it is tiled; only its epilogue and prologue fuse.                                                                                             |Q-E            
Tiling and order of addition |Tiling decides where operands are read from; the accumulation loop decides the order, so a tiled matmul can keep split-K's bits.                                       |Q-X (open)     
Online softmax               |A running maximum with rescaling by `exp(m − m')` lets softmax work tile by tile, as in FlashAttention.                                                                |Q-G            
Associativity and determinism|Float addition order changes the bits. Levels 1, 2 and 3 of Decision K.                                                                                                |Q-H, Q-I       
Prefix caching               |Why level 2 matters: a cache hit must give the same bits as recomputing.                                                                                               |Q-I            
Auto-vectorisation           |Loops over `Slice` vectorise only without per-element bounds checks or opaque calls (libm); float sums do not reorder.                                                 |none           
Loop-carried dependencies    |Vectorise across independent outputs; a running total can't be split without reordering.                                                                               |Q-O            
Quantised dot products       |Q4 × Q8 in integers vectorises and is exact within a block.                                                                                                            |none           
Where BLAS's speed comes from|Bookkeeping, loop order, register and cache blocking, threads, matrix units: factors that multiply.                                                                    |Q-N            
Decode is memory-bound       |Each token streams all the weights; the GPU's edge shrinks on unified memory.                                                                                          |none           
NVIDIA driver packages       |CUDA is in `libnvidia-compute`, Vulkan in `libnvidia-gl`; the container toolkit mounts the host's, so a container cannot add what the host lacks.                      |Q-AF           
Linking and loading          |`@[Link]` makes a library a dependency at build and at start; `dlopen` defers both. Tensomite links (Decision AV).                                                     |none           
Fused multiply-add           |One rounding instead of two: more accurate, but different bits from the host, so CUDA kernels use `--fmad=false`.                                                      |Q-AG (open)    
Reverse-mode autograd        |`grad` is a graph transform: chain rules build a backward graph of ordinary ops; views' adjoints sum (`Unview`).                                                       |not yet        
Split-K and determinism      |Fixed-size chunks of `k`, combined in chunk order, keep bits independent of row count and zero-padding; costs ~0–8%.                                                   |Q-R            
Fused kernels and rounding   |A fused CPU loop rounds exactly as the composite does, so the composite is a bit-exact oracle; GPU `tanh` differs in the last bit.                                     |Q-S (explained)
Finalizers and threads       |A finalizer can interrupt its own thread mid-allocation, so it must never wait for a lock that thread may hold.                                                        |Q-U            
GPU lifetimes                |On Metal, "its consumer ran" means encoded, not executed; the CPU must not reuse a buffer the GPU has yet to read.                                                     |Q-V (explained)

Questions asked but never answered: Q-D (why a SPIR-V emitter in Crystal needs no compiler at run time), Q-J (why `sum(axis)` permutes the axis to the end), Q-L (why `Cat` writes through a slice view), Q-M (why `max` starts from the first element), Q-P (why broadcasting needs no gradient rule of its own), Q-Q (the gradient of `x` expanded from `[3]` to `[2, 3]`), Q-R (why the split must depend only on `k` for prefill to equal decode), Q-T (why softmax's separate `exp` pass keeps the bits and vectorises), Q-W (what `b`'s gradient looks like in a diagram, given its `expand`), Q-X (why a 4×4 patch per thread cuts memory reads), Q-Y (how named parameters simplify the decay split). Q-K was answered, after a correction: a stride of 0 means repeated reads of one position, not contiguous rows.
