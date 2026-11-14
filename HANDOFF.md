# Tensomite: handoff

What the next session, or the next person, needs in order to carry on without the history of the last one. `PLAN.md`, `DECISIONS.md`, `DESIGN.md` and `DEVELOPMENT.md` hold the project itself. This document holds how we work, where things stand and what has been learned.

**Keeping it current.** Rewrite it rather than appending to it: at the end of each PR, and whenever a session's context passes about 70%. Anything that belongs in one of the other four documents goes there instead.

*Last rewritten: 14 Nov 2026, with M11 met and Decision DG settled.*

## Where things stand

- **On `main`:** milestones 1 to 11, except M8's exit run on a discrete GPU (a T4), which waits for one. The lazy tensor core, autograd and training, and five devices: `CPU` (the oracle), `SimDiscrete`, `BLAS`, `Metal` and `CUDA`. Models on `Tensomite::Module`s with a `Builder` DSL; GPT-2 and Llama-family models (SmolLM2) from safetensors and GGUF; block formats `Q8_0`, `Q4_0` and `Q4_1` with ggml's bytes; `apps/server` and `apps/finetune`. Evidence for each exit is under its milestone in `PLAN.md`.
- **M10, quantisation (Decisions CL to CM):** met 9 Nov 2026. Quantised products have kernels on every device, with the host's bits on CUDA.
- **The GPUs after M10 (Decisions CN to DC):** fused RMSNorm and RoPE, wider quantised products on CUDA (`__dp4a`), fused attention on GPUs for decoding's few rows. SmolLM2-1.7B at Q4_0 decodes in 5.71 ms a token on the GB10 and 6.60 on the M5 Pro; llama.cpp takes 4.85 and 4.73. Metal's quantised products are the largest gap left (Decision CX); CUDA graphs were deferred (Decision DC).
- **M11, fast quantised CPU kernels (Decisions DD to DF):** met 14 Nov 2026. A CPU decode profiler (`bench/profile_decode_cpu.cr`); `Quant.matmul`, with `Quant.matmul_reference`'s bits; a spinning thread pool (`TENSOMITE_THREADS`); RoPE along strides; the serial work taken out of each product; signed nibbles, so native builds get `sdot`; two weight rows per pass. SmolLM2-1.7B at Q4_0 decodes on the CPU in 7.17 ms a token on the M5 Pro (6.36 native) and 12.8 on the GB10 (11.2 native); llama.cpp takes 6.7 and 11.4. Native builds are opt-in (`ops bn`, `ops run-native`; Decision DE).
- **Now (Decision DG):** the documents, issues #114 to #120: this rewrite and the other stale parts first, then a glossary (#118), then the measurement history and decisions moved into files of their own (#119). Then generation and the server: #121 (the head on the last row only in prefill), #122 (the sampler's sort, and a bench that samples) and #127 (one `Cache` and one `generate`).
- **A long-term goal:** inspecting running models (activations, attention, gradients) by the names of their parts; see "Inspectable" in `PLAN.md` and "Towards inspection" in `DESIGN.md`. Scopes are its first piece, and Decision AA must keep inspected nodes alive.
- **Deferred:** CUDA graphs (Decision DC: decode is GPU-bound on the 1.7B model); hand-written `sdot` in inline assembly (Decision DE; LLVM emits it from signed bytes, Decision DF); Decision AA for inference, whose buffers return through the collector, while training steps release them as they run (Decision CD); Vulkan (M12), until a machine with a graphical driver is at hand (Decision AT).

## Working agreements

Agreement           |Practice                                                                                                                                                                                                                                                                            
--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Branches            |One per PR, named for its purpose (`m5-autograd`, `docs-handoff`), cut from the latest `main`.                                                                                                                                                                                      
Pull requests       |Claude opens them without waiting to be asked, then watches them: CI, reviews and merge, with a safety-net check-in.                                                                                                                                                                
Stacked PRs         |When needed. The later PR's base is `main`; its description names the PR it stacks on, and its diff includes that PR's until it merges, after which `main` is merged into it.                                                                                                       
Merging             |The maintainer merges, by squash or by merge commit. After a squash, merge `main` into a stacked branch and resolve conflicts with the branch's side; never rewrite history or force-push (the session blocks it anyway).                                                           
Milestones          |Each ends with its exit criterion passing, with the evidence (numbers, logs) in the PR description.                                                                                                                                                                                 
Documents           |Updated in the same PR as the code (see "Workflow" in `DEVELOPMENT.md`), `GUIDE.md` included whenever the public API changes. Tables and diagrams follow "Documents" in `DEVELOPMENT.md`; check them with `scripts/format_tables.py` and `scripts/check_diagrams.sh`.               
Decisions           |Labelled with letters, recorded in `DECISIONS.md` and indexed in `PLAN.md`. The next is **DM**. P (how to get GPT-2 small's fixtures) was settled by allowing Hugging Face; Q made this document committed.                                                                         
Understanding checks|Short questions labelled Q-A, Q-B and so on, after explaining something new. The next is **Q-CK**. Each is written up in `faq/`, a page per question, grouped by theme (Decision DH).                                                                                               
Delegation          |Mechanical tasks go to the `implementer` subagent (`.claude/agents/implementer.md`, Sonnet) with a brief of the files, the change and its acceptance; the main session reviews the diff, reruns the checks, commits, and adds any lesson to the agent's brief (Decisions DK and DL).
Claims              |Measured before stated. Several assumptions were wrong when checked: decode time was dominated by the KV cache, not host kernels; GPT-2's file was unaligned. Say "I think" when something is unverified.                                                                           

## Follow-ups found on the way

Item                               |Why                                                                                                                                                                                                                                                   |Where it fits                                         
-----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------
A declarative model macro          |`ln1(x)` rather than `@ln1.call(x)` in `forward`; the macro tried in M7b (`parts ... do ln1 = layernorm(width) end`) made ameba report each part as a useless assignment, so a new one needs a syntax ameba accepts without suppressions (Decision AR)|When a model's `forward` grows long                   
Narrow Metal matmuls               |Products with 768 output columns run at about 950 GFLOP/s, Accelerate's half; splitting `k` across threadgroups in split-K's order means storing each chunk's sum (Decision AG)                                                                       |When prefill matters again                            
Hardware matrix instructions       |`simdgroup_matrix` (and the M5's GPU matrix units, I think) could beat the tiled kernel, but their order of addition is unspecified                                                                                                                   |An opt-in mode with relaxed determinism, like `Atomic`
Planned buffer release in inference|Lower peak memory in a forward pass, as Decision CD gave training steps (Decision AA)                                                                                                                                                                 |With the server's batching                            
Fewer passes over memory           |A host pass costs 0.7 ns per element in training; the remaining fusions (attention's scale and mask, AdamW, bias adds) would save some                                                                                                                |Optional (Decision Z)                                 
Parallel host kernels              |Quantised products run on all cores (M11); training's elementwise passes and `F32` matmuls outside BLAS still run on one                                                                                                                              |When CPU training matters                             
`Int32` element counts             |Caps one tensor at 2 GiB                                                                                                                                                                                                                              |Before models with larger tensors                     

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

**Training data:** `scripts/fetch_shakespeare.sh` (from GitHub raw, allowed by default); `scripts/make_code_corpus.sh` for Crystal's standard library, from the installed compiler.

**High `sys` time** in training is OpenBLAS's threads busy-waiting, not the collector; `OPENBLAS_NUM_THREADS=1` removes it.

**Specs run in random order** (`ops test_specs`); a run ends with `Randomized with seed: N`, and `crystal spec --order N` replays it.

**Long runs.** Start them with `nohup … &` and a log in the scratchpad, then wait with a background `until grep -q … log; do sleep 30; done`; a foreground `sleep` is blocked. The container is reclaimed when the session goes idle, which kills background jobs: a run of several hours survives only while the session is active, so run those on the maintainer's machine.

**Shell gotchas.** `rm` on a path built from a shell variable (`rm -f "$D"/*`) is blocked; use a literal path or `"${D:?}"`. Force-pushes are blocked. Crystal cannot `require` an absolute path (see "Crystal gotchas" in `DEVELOPMENT.md`), so temporary scripts that load tensomite go inside the repo and are deleted afterwards.

### Maintainer's machine

A MacBook Pro with an M5 Pro, using Accelerate, and Homebrew's Crystal 1.22.2. Its LLVM rejects some intrinsics that CI's Crystal 1.22.0 accepts, such as `llvm.convert.from.fp16`; a cross-compile in the container doesn't link, so it can't catch that, and only a build on the Mac can. Its current benchmark results are in "Performance baselines" in `DESIGN.md`. A Python venv there must also use 3.13 or older until PyTorch supports newer releases.

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
Amdahl's law                 |Work on one thread bounds what more threads can buy: 2 ms serial of a 15 ms token caps 20 threads at 2.65 ms. Found by timing before changing.                         |none           
Signed bytes and `sdot`      |LLVM emits `sdot` only when both operands are signed bytes; unpacking nibbles less 8 gives the same integer and lets it.                                               |none           
Tables against arithmetic    |A lookup in a table whose used entries stay in cache can beat a few more instructions, when it runs once per 32 weights.                                               |none           

Questions asked but never answered: Q-D (why a SPIR-V emitter in Crystal needs no compiler at run time), Q-J (why `sum(axis)` permutes the axis to the end), Q-L (why `Cat` writes through a slice view), Q-M (why `max` starts from the first element), Q-P (why broadcasting needs no gradient rule of its own), Q-Q (the gradient of `x` expanded from `[3]` to `[2, 3]`), Q-R (why the split must depend only on `k` for prefill to equal decode), Q-T (why softmax's separate `exp` pass keeps the bits and vectorises), Q-W (what `b`'s gradient looks like in a diagram, given its `expand`), Q-X (why a 4×4 patch per thread cuts memory reads), Q-Y (how named parameters simplify the decay split), Q-BU (why the output layer reads the most bandwidth), Q-BV (why 256 threads with 112 idle beat 36), Q-BY (what exactness buys against a few percent of speed), Q-CA (why fused attention matters more as the context grows), Q-CB (why repeating a quantising pass per threadgroup could be faster), Q-CC (what a launch-bound profile looks like), Q-CE (why sums within a block can split across lanes but sums over blocks can't), Q-CG (whether portable macOS builds should enable `+dotprod`, given who sets the flags when tensomite is a shard). Q-K was answered, after a correction: a stride of 0 means repeated reads of one position, not contiguous rows.
