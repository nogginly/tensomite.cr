# Tensomite: decisions

Every design decision, labelled with letters in the order they were taken, with its reasoning and the measurements behind it. `PLAN.md` indexes them in a line each. A decision that a later one revised says so, and keeps its text: the reasoning is the point.

Three labels have no entry. P (how to get GPT-2 small's fixtures) was settled by allowing downloads from Hugging Face, and Q by committing `HANDOFF.md`. BX, a choice of what to build next, was raised in a session and never recorded.

<a id="decision-a"></a>
## A: own backends or bind ggml

Own backends (decided).

<a id="decision-b"></a>
## B: Linux GPU

Vulkan (decided), then superseded by Decision AT: CUDA first, Vulkan as M12. Vulkan still covers AMD, Intel and NVIDIA through one device.

<a id="decision-c"></a>
## C: first model

GPT-2 small, then TinyLlama (decided).

<a id="decision-d"></a>
## D: CPU parallelism

`-Dpreview_mt` or execution contexts. Defer until profiling shows the CPU path matters.

<a id="decision-e"></a>
## E: execution model

Lazy graph with `realize` (decided).

<a id="decision-f"></a>
## F: tensor typing

Non-generic `Tensor` with a runtime `DType` (decided).

<a id="decision-g"></a>
## G: Metal plus Vulkan, or Vulkan everywhere via MoltenVK

Native Metal (decided). Revisit after M8, when keeping two codegen emitters has a known cost.

<a id="decision-h"></a>
## H: Vulkan shader compilation

Compile at build time (decided). A Crystal macro runs `glslc` during `crystal build` and embeds the SPIR-V, so users need no compiler and there is no shaderc seam.

<a id="decision-i"></a>
## I: fused kernels on Vulkan

Settled by Decision J. Macro kernels emit GLSL at compile time, from M8 (Decision V).

<a id="decision-j"></a>
## J: how kernels are fused

Compile-time kernel macros (decided). Fusion is explicit: only what is written inside one macro is fused. Runtime auto-fusion over the graph is added only if measurements after M8 show a need for it. That would bring back the shader-compiler question for Vulkan (shaderc or a SPIR-V emitter in Crystal).

<a id="decision-k"></a>
## K: determinism

Three levels (decided).
  1. Run to run: the same inputs, shapes and device give the same bits. Guaranteed everywhere, including parallel CPU code, which splits work independently of thread count.
  2. Shape invariance: a position's logits do not depend on batch size or prompt chunking. A goal for tensomite's own inference kernels, tested by exact comparison of prefill and decode. It makes prefix caching and chunked prefill reproducible: a cache hit gives the same bits as recomputing. The BLAS backend is exempt, because vendor libraries choose strategies by shape.
  3. Across devices: never promised. Comparisons use tolerances.

<a id="decision-l"></a>
## L: level 2 on CPU

Devices report `shape_invariant?` (decided, lands in M3). BLAS reports false, and features that need level 2 fall back to tensomite's own matmul or warn. A cache-blocked CPU matmul in Crystal, shape-invariant by construction, is an optional later milestone if CPU inference matters.

<a id="decision-m"></a>
## M: a fast CPU path

Own quantised CPU kernels after quantisation (M10) (decided), as milestone 11. Quantised small models make the CPU worth running fast, and BLAS cannot help with them: `sgemm` takes `f32` only. The oracle stays naive.

<a id="decision-n"></a>
## N: modelling dtypes for quantisation

Settled early so quantisation (M10) adds code rather than refactoring it (decided).
  1. Byte sizes come only from `DType#bytes_for(count)`, never from a per-element size.
  2. Each op declares the dtype combinations it accepts. Today every op requires one dtype for all sources. From M10, `MatMul` accepts a block-format weight with `F32` activations and output, and `Gather` a block-format table. Host kernels then dispatch on the source dtypes as well as the output dtype.
  3. Block-format tensors are leaves loaded from files, stored row by row, sliced only by whole rows, and never produced by an op. `View` stays in logical elements.
  4. `F16` and `BF16` are tensomite's own types, stored as `UInt16` and computed in `Float32`.

<a id="decision-o"></a>
## O: when Metal must beat BLAS

Measured in M6, required in M7 (decided). On Apple Silicon, Accelerate reaches about 1 TFLOP/s on a 512 matmul (M5 Pro), and GPT-2 decoding is a run of small matrix-vector products where a GPU's dispatch overhead and the shared memory bandwidth erode its advantage. Fewer dispatches per token, from kernel macros, are what is expected to give the GPU the lead.

<a id="decision-r"></a>
## R: the M5 training exit

Below 1.8 nats/char within an hour in M5, and below 1.6 within an hour in M7 (decided). Training works: the 0.8M-param model reaches 1.80 by step 2,250 (27 minutes) and 1.726 after 3,000 steps (36 minutes) in the 4-core container. Reaching 1.6 needs a larger model and a longer context, about 1.6 s a step and two to three hours here, because training is bound by elementwise passes. Fusion and buffer reuse in M7 attack exactly that, so the 1.6 target becomes a measure of what they buy. The alternatives were a multi-hour run, which an idle cloud container does not survive, and bringing Decision D (CPU threads) forward into M5, which deserves its own design for deterministic work splitting.

<a id="decision-s"></a>
## S: Metal matmul against the CPU's bits

Metal's default matmul splits `k` across threads, so it may differ from the CPU in the last bits (decided). Level 3 of Decision K never promised otherwise. Levels 1 and 2 still hold: each output's sum is cut into chunks at fixed offsets and combined in chunk order, so the bits repeat from run to run and do not depend on the number of rows or on zero-padding of `k`. A serial mode, bit-identical to the CPU, remains for checking.

<a id="decision-t"></a>
## T: how row ops are fused

`softmax`, `log_softmax` and `layernorm` are hand-written fused ops, with the existing composites as their oracles and hand-derived backward rules checked by finite differences (decided). Kernel macros stay elementwise. Revisit after M8 if more row ops appear.

<a id="decision-u"></a>
## U: how buffers are reused

A pool fed by finalizers (decided). When the collector frees a buffer, its memory joins a free list by size, and allocation takes from that list first. On Metal, a buffer re-enters the pool only after the command buffer that last used it has completed. Freeing intermediates at their last use within a realisation would be tighter, but is unsafe while a user can hold a `Tensor` over any node.

<a id="decision-v"></a>
## V: when kernel macros emit GLSL

In M8, when a Vulkan device can run it (decided). Emitting code that nothing runs invites drift. The emitter is designed with GLSL as a third backend.

<a id="decision-w"></a>
## W: if M7's training is still too slow

Decided after the fused models are measured in M7. The next lever would be Decision D, CPU threads, with work split independently of thread count.

<a id="decision-x"></a>
## X: Metal reductions against the CPU's bits

Metal's `Sum` of a long run folds it in chunks of 64 in parallel, so it may differ from the CPU in the last bits (decided). The rules of Decision S apply: chunks sit at fixed offsets and combine in chunk order, so bits repeat from run to run and do not change when a row is padded with zeros. `Max` is exact in any order, and runs of one chunk keep the host's arithmetic.

<a id="decision-y"></a>
## Y: transcendental functions on host devices

Tensomite's own `Float32` `exp`, `log` and `tanh` (`Tensomite::Maths`), not the C library's (decided). A call into libm cannot be vectorised, and `tanhf` alone cost 12 ns per element, most of `gelu`'s time in training. Each function is range reduction and a polynomial written with selects, within 2 ulp of the correctly rounded result. They pay only in loops that vectorise, so host loops read unit-stride runs through pointers after checking each run's bounds once. Together they halve a training step. Host results no longer depend on the platform's maths library, so Linux and macOS give the same bits; they differ from libm's in the last bits, within the fixtures' tolerances.

<a id="decision-z"></a>
## Z: after part 6's speed-up

Run the training exit at once, and treat the remaining fusions as optional (decided). It passed, so Decision W's CPU threads are not needed for M7.

<a id="decision-aa"></a>
## AA: planned release of intermediate buffers

After M7 part 8 (decided; done for training steps by Decision CD, still open for inference). The scheduler would return a node's buffer to its pool once its last consumer has run, rather than when the collector finalizes it, which would lower a step's peak memory and make buffer lifetimes independent of the collector and its threads. Only nodes internal to the realised graph qualify, since a tensor the caller holds must stay valid. On Metal a buffer would be released when its consumer's command buffer completes, not when it is encoded, because the CPU may write a reused buffer before the GPU has read it.

<a id="decision-ab"></a>
## AB: a Mermaid view of the graph

`Tensor#to_mermaid`, as its own PR between M7 parts 6 and 8 (decided). Node ids follow topological order, so the same graph gives the same text.

<a id="decision-ac"></a>
## AC: model summaries

Scopes (`Tensomite.scope`) and summaries by scope in the same PR as Decision AB (decided). Scopes are designed for inspection too: each call makes a new instance, so one of several repeated blocks stays addressable, and gradient nodes take the scope of the node they differentiate.

<a id="decision-ad"></a>
## AD: how Metal's matmul is tiled

Blocks of outputs per threadgroup, operands shared through threadgroup memory, adding in split-K's exact order (decided). Tiled and split products then have the same bits, so prefill and decode stay bit-identical (Decision K, level 2), and the choice between them is speed alone. Apple's 8×8 matrix instructions, whose order of addition is unspecified, are ruled out; they could return as an opt-in mode with relaxed determinism.

<a id="decision-ae"></a>
## AE: matmul epilogues

Dropped from M7 (decided). Fusing the bias adds into the matmul would save under 1% of a prefill's GPU time and about 5% of a decoded token's.

<a id="decision-af"></a>
## AF: part 8's target

Metal prefill faster than BLAS in the same run, with decode no slower (decided, met).

<a id="decision-ag"></a>
## AG: after the target was met

Stop tuning; products with 768 output columns stay slower than Accelerate (decided). Speeding them up needs split-K across threadgroups, and keeping split-K's order then means storing every chunk's sum.

<a id="decision-ah"></a>
## AH: how models are defined

Modules in plain Crystal first, then a DSL that builds on them, iterated together (decided). The DSL may need no macros, or may not be needed at all once the modules settle.

<a id="decision-ai"></a>
## AI: how parameters change

A mutable `Parameter` slot holding an immutable tensor (decided). Optimizers replace each slot's tensor, so training loops do not reassemble parameter lists. The alternative, functional modules returning updated copies as in JAX, keeps everything immutable at the cost of rebuilding the model every step.

<a id="decision-aj"></a>
## AJ: how parts are found

Instance variables listed at compile time, named after the variables (decided), rather than explicit registration calls.

<a id="decision-ak"></a>
## AK: initial weights

Each module takes an `rng:`, defaulting to `Tensomite.rng`, seeded, so runs repeat unless asked otherwise (decided). `Linear` follows PyTorch (uniform, ±1/√inputs) and `Embedding` GPT-2 (normal, 0.02).

<a id="decision-al"></a>
## AL: where modules sit in the plan

Milestone 7b, before Vulkan, so that later milestones keep their numbers (decided).

<a id="decision-am"></a>
## AM: saving

A safetensors writer, so models save and load in the format GPT-2 already loads from (decided).

<a id="decision-an"></a>
## AN: the model DSL

A `Builder` in plain Crystal now; a declarative macro reconsidered after GPT-2 is on modules (decided). The macro tried in M7b part 4 (`parts(...) do ln1 = layernorm(width) end`) gave `forward` the shorter `ln1(x)`, but ameba flags each part line as a useless assignment. A macro that returns must pass ameba without suppressing any rule.

<a id="decision-ao"></a>
## AO: names of GPT-2's parts

Hugging Face's, in the modules themselves (decided): `CausalSelfAttention` holds `c_attn` and `c_proj`, and `GPT2` holds `wte`, `wpe`, `h`, `ln_f`, with `ln_1`, `attn`, `ln_2` and `mlp` (`c_fc`, `c_proj`) in each block. The file then loads by name with no table of renamings, and diagrams use the file's names.

<a id="decision-ap"></a>
## AP: loading files with other tensors or prefixes

`Module#load(prefix:, ignore:)` (decided), so strict loading still catches real mismatches while skipping stored masks such as `attn.bias` and adding prefixes such as `transformer.`.

<a id="decision-aq"></a>
## AQ: where the KV cache lives

One `CausalSelfAttention::Cache` per block, passed to `call` (decided). `GPT2` keeps its API: `load`, `new_cache`, `forward(ids, cache)` and `generate`.

<a id="decision-ar"></a>
## AR: closing 7b

The `Builder` stands as the model DSL, and the declarative macro's second look is a follow-up rather than part of 7b (decided). The builder removed the repetition that prompted a DSL; what a macro would still add is `ln1(x)` for `@ln1.call(x)` in `forward`.

<a id="decision-as"></a>
## AS: where M8 runs

Developed on the maintainer's DGX Spark, with the exit on a cloud T4 (decided). The Spark's GPU shares its memory with the CPU, but the device copies explicitly between host and device memory anyway, so the Spark exercises the discrete path and the T4 proves it on a PCIe card.

<a id="decision-at"></a>
## AT: the first Linux GPU backend

CUDA, ahead of Vulkan (decided). Both machines are NVIDIA's and already run CUDA, while the Spark's compute-only driver lacks Vulkan's part (`libnvidia-gl`). AMD and Intel GPUs wait for Vulkan, M12.

<a id="decision-au"></a>
## AU: how CUDA kernels are compiled

By NVRTC when first used, as Metal compiles MSL (decided). Building needs no CUDA toolkit, running needs `libnvrtc` beside the driver, and CI compiles every kernel without a GPU. Kernels are compiled with `--fmad=false`, so `a * b + c` rounds as on the host. PTX is built for the newest architecture NVRTC knows at or below the GPU's, and the driver compiles it for the GPU.

<a id="decision-av"></a>
## AV: how the CUDA libraries are reached

Linked at build time behind `-Dcuda` (decided), as BLAS and Metal are, rather than loaded with `dlopen`. A program is built for the machine it runs on; one built with `-Dcuda` needs the driver and NVRTC to start.

<a id="decision-aw"></a>
## AW: CUDA's decode speed

Profile before changing anything, then fix the largest cost (decided). On the GB10, GPT-2 small decodes at 171 tokens/s, about three times the floor its memory bandwidth sets, as Metal on the M5 Pro does; the suspects are host overhead per launch, the output head reading its weights through a transpose, and small synchronous copies each token. The preallocated KV cache is a separate change, for every device.

<a id="decision-ax"></a>
## AX: the first decode fix

A two-pass matmul for products of fewer than 16 rows, bit-identical to `Split` (decided). The profile put 3.8 of a decoded token's 6.4 ms in matmuls reading weights at about 129 GB/s, the output head at 94: `Split` launches too few blocks for one row and reads the transposed head across its rows. The first pass sums each chunk of 64 along `k` in its own block, with the weights staged in shared memory along their contiguous axis; the second adds the chunk sums in chunk order. Estimated 4.8 ms a token, 210 tokens/s; measured 5.0 ms, 198 tokens/s, all of the gain from the head.

<a id="decision-ay"></a>
## AY: which products get two passes

Only those whose `b` has non-contiguous columns, as GPT-2's output head does; `Split` keeps the rest (decided). Two passes made the head 2.2 times faster but slowed qkv and attention out, whose reads already coalesced under `Split`, and added a launch per matmul. Estimated about 4.7 ms a token; the profile measured 4.71, about 212 tokens/s. Reworking the two-pass kernel for contiguous weights is left for later.

<a id="decision-az"></a>
## AZ: decode's remaining costs

Measure GPU time per op whatever the shape, then fix the largest (decided). Each small kernel costs about 4.6 µs however little it does, and the KV cache's `Cat` is split across many shapes in the profile, so its total is hidden.

<a id="decision-ba"></a>
## BA: AMD GPUs

Vulkan as M12, then HIP, ROCm's CUDA-like API, as a later and smaller milestone (decided). HIP mirrors the CUDA driver API and has a run-time compiler, `hiprtc`, so a HIP device would be largely a port of the CUDA one; Vulkan reaches GPUs ROCm does not support. Revisit when an AMD machine with ROCm is at hand.

<a id="decision-bb"></a>
## BB: the first of decode's remaining costs

A preallocated KV cache, written in place, then a few-row kernel for contiguous weights (decided). The cache's `Cat` was 0.52 ms of a decoded token's 4.75 ms of GPU time on the GB10, copying each layer's whole cache to add one position; writing in place helps every device and keeps shapes fixed, which CUDA graphs would need. It is done with `Tensor#write`, the one op that changes a buffer, rather than by making tensors mutable in general. Measured: `Auto` decodes at 246 tokens/s on the GB10, from 204, and `BLAS` at 94.6, from 71.2.

<a id="decision-bc"></a>
## BC: how the KV cache is allocated

Its whole capacity on the first append (decided), accepting about 1.6 ms more before the first token on the GB10: 1.3 ms allocating the buffers and 0.4 ms copying the prompt into them. Growing by doubling would avoid that, but a doubling briefly holds the old buffer beside one twice its size, and frees it only when the collector runs (Decision AA). That matters for larger models on smaller GPUs, where the cache, not GPT-2's 75 MB, is gigabytes; a fixed allocation also keeps memory predictable and shapes fixed.

<a id="decision-bd"></a>
## BD: after the cache

The few-row kernel for contiguous weights, as Decision BB planned, then perhaps bias adds fused into matmuls (decided). The four contiguous decode products read 339 MB a token at 136–200 GB/s; at 210 to 240 GB/s they would save 0.27 to 0.47 ms of 4.0. The kernel is split-K with narrower blocks, so that a product with few columns has enough of them, keeping its bits.

<a id="decision-be"></a>
## BE: after narrower blocks were slower

Keep 32-column split blocks, drop the narrower ones and keep their benchmark (decided). On the GB10, 16- and 8-column blocks read the decode weights at 71–202 GB/s against 160–209 for 32 columns; 32 columns already reads at about 200 GB/s alone, so the few-row kernel has about 0.15 ms a token left to win, not 0.27–0.47.

<a id="decision-bf"></a>
## BF: decode's next costs

The 48 copies of a decoded token, then bias adds fused into matmuls, then part 5 (decided). The copies were `reshape`s of views that strides could express, so `reshape` now returns a view whenever strides allow, as PyTorch's `view` does; the bits are unchanged. Bias adds became a third source of `MatMul`: a contiguous `[N]` bias, which each kernel adds after its sum as `bias[col]`, with the bits of a separate `Add`; any other bias is added after the product. A first version read the bias through a general view, with integer divisions per output, and cut prefill on the GB10 by a third. Measured with both changes: `Auto` decodes at 268 tokens/s on the GB10, from 246, and prefills 64 tokens at 5,050 tokens/s, from 4,134.

<a id="decision-bg"></a>
## BG: CUDA training speed

Profile a training step, then fix its largest cost (decided). On the GB10 the larger Shakespeare model trains at 110 ms a step, slower than Metal's 76 on the M5 Pro though the GB10's GPU is the larger; decode on the GB10 already beats the M5 Pro. The profile put a third of the GPU time in `Unview`, whose index arithmetic was rewritten, and 17 ms of host time in allocation, which was missing the buffer pool because finalizers ran inside one another and freed their memory; and kernels copied their view arguments to per-thread local memory, which unrolled loops now avoid. With the first two fixes a step took 83.4 ms on the GB10, and Metal, which shared the pool bug, 59 on the M5 Pro, from 76; with the third, about 20 ms on the GB10.

<a id="decision-bh"></a>
## BH: profiling Metal training

`--profile` for `--device metal` in the trainer, sharing the report with CUDA (decided). `ProfileReport` prints the tables for both devices from the parts each one measures; `Metal::Profile` gained the pool's counters, and `bench/profile_decode.cr` prints the shared report. With CUDA at about 20 ms a step on the GB10 and Metal at 59 on the M5 Pro, the profile is to show where Metal's time goes. It put 37% of the GPU time in `Unview` and 7% in `ScatterAdd`, and left 17 ms of a 59 ms step outside any op.

<a id="decision-bi"></a>
## BI: Metal training speed

Port CUDA's `Unview` to Metal, and zero `ScatterAdd`'s output in parallel on both GPUs, with the same bits; then, in a separate PR, find why a step commits 114 command buffers rather than about 14 (decided). `Unview [192]` took 0.83 ms a call on the M5 Pro, as it took 0.74 on the GB10 before Decision BG, and a one-column `ScatterAdd` zeroed 133,120 elements on one thread.
- **Decision BI's second part, the waits in a training step:** the 114 command buffers came from the optimizer, not from Metal: `Scheduler.realize` waits for the device, and AdamW detached `m`, `v` and each parameter's new value separately, so a step waited three times per parameter and realised the backward pass a piece at a time. The optimizer now realises the loss, its state and every update with one `Tensor.realize`, with the same bits. On the M5 Pro a step commits 15 command buffers, and takes 40.0 ms from 59.1 without the first part; CUDA should gain too, since it waited as often.

<a id="decision-bj"></a>
## BJ: after Decision BI

First a dense path in Metal's `Unview` that loads 32 elements ahead of adding them, in order; then, in a separate PR, long-`k` products with few output tiles split across blocks along `k`, with split-K's bits, on both GPUs (decided). With both parts of Decision BI, the larger Shakespeare model trains at 30.7 ms a step on the M5 Pro and 20.5 on the GB10. `Unview [192]` takes 0.40 ms a call on Metal and 0.032 on CUDA with the same algorithm, I think because Metal waits for each load in turn; weight gradients such as `[192,2048]·[2048,192]` run at 330 and 468 GFLOP/s, in nine tiles of 64 by 64, against 2,100 to 5,300 for forward products. Measured: eight loads ahead took `Unview [192]` to 0.058 ms a call, and 32 to 0.027, with a Metal step at 25.9 ms; 1, 8 and 32 were measured, with threadgroups of 256 and 32. The second part is `MatMulMode::TiledTwoPass`, which `Auto` picks for fewer than 64 blocks of outputs and at least four chunks of `k`: a training step takes 22.4 ms on the M5 Pro, from 25.9, and 17.2 on the GB10, from 20.5.

<a id="decision-bk"></a>
## BK: after Decision BJ

Share the scalar constants arithmetic with numbers makes, by device, dtype, value and scope; then speed up cross-entropy's gradient, its `ScatterAdd` into one column (decided). With `TiledTwoPass`, a training step uploaded 363 constants on the GB10, 1.26 ms of synchronous copies, and `ScatterAdd [133120,1]`, a single-thread chain of 2,048 adds, took 0.66 ms on the M5 Pro and 0.82 on the GB10.
- **Decision BK's second part, cross-entropy's gradient:** a `ScatterAdd` into one column runs a thread per id rather than one thread in all, with the same bits; no change to autograd. Its gradient was `ScatterAdd [133120,1]`, 0.66 ms on the M5 Pro and 0.82 on the GB10; it now takes about 0.1 on each. With both parts on `main`, a training step of the larger model takes 21.8 ms on the M5 Pro and 15.0 on the GB10, from 59.1 and 20.5 before Decision BI.

<a id="decision-bl"></a>
## BL: after Decision BK

An FAQ of the understanding checks, then fusing elementwise ops; M8's exit runs wait for a T4 (decided). The FAQ is `faq/`, a page per question in ten themes.

<a id="decision-bm"></a>
## BM: which elementwise ops to fuse first

AdamW's, with kernel macros, three per parameter (decided). A training step of the larger model launched 925 kernels, over half of them AdamW's primitive ops, about fifteen for each of 36 parameters. Its step becomes three macro kernels, for `m`, `v` and the new value, whose per-step numbers are scalar inputs; each does the composite's operations in its order, so the bits are unchanged. Measured: 511 launches a step, with the same bits; a step takes 20.8 ms on the M5 Pro, from 21.8, and 14.1 on the GB10, from 15.0. Runtime fusion over the graph (Decision J) stays unneeded: the remaining launches are spread over the forward and backward passes, where attention's scale and mask are the next candidates.

<a id="decision-bn"></a>
## BN: after the fused AdamW

A fast path for the inputs of kernel macros and elementwise ops, then attention's scale and softmax's backward, then layer norm's backward (decided). A fused kernel on `[192]` took 4.7 µs against 2.8 for a `Mul`, I think from finding each of its five broadcast scalars by integer divisions per element. Those kernels now receive a view without its leading dimensions of stride 0, which add nothing to a position: a broadcast scalar becomes a view of no dimensions, read with no arithmetic. The bits are unchanged. Measured: 20.5 ms a step on the M5 Pro, from 20.8, most of it from attention's scale by a scalar; no change beyond noise on the GB10, whose elementwise ops were already bound by memory.

<a id="decision-bo"></a>
## BO: attention's softmax

Fuse the scale and mask into `Softmax`, and softmax's backward into one op, `SoftmaxGrad`, with the bits of the primitives they replace (decided). Per layer of the larger model, attention's scores `[16,6,128,128]` went through a `Mul` by the scale and an `Add` of the mask before the softmax, and through five ops after it in the backward pass: 16 `Mul`s, 4 `Sub`s and 4 `Sum`s a step, and on the GB10, which is bound by memory for these, about ten passes over 6.3 MB per layer. `SoftmaxGrad` adds `g · y` in the order `Sum` does, so the losses keep their bits. On the CUDA stub a step launches 487 kernels, from 511, and uploads 25 constants, from 29. Measured: 19.6 ms a step on the M5 Pro, from 20.5, and 13.6 on the GB10, from 14.3; estimated for the GB10 from its traffic, 0.9 ms.

<a id="decision-bp"></a>
## BP: layer norm's backward

One fused op, `LayerNormGrad`, for the input's gradient, and the same op with `x` alone for `x̂`, which the weight's gradient multiplies by `g` (decided). The composite recomputed each row's statistics and took about sixteen primitives over `[2048,192]` per layer norm, nine layer norms a step; most of the step's remaining `Mul`, `Sub`, `Div` and `Sum [2048]` launches were these. Each row's sums are added in `Sum`'s order, so the losses keep their bits. On the CUDA stub a step launches 334 kernels, from 487, and uploads 7 constants, from 25. Measured: 18.7 ms a step on the M5 Pro, from 19.6, and 12.9 on the GB10, from 13.6; `LayerNormGrad` takes about 37 µs a call on both, about 0.7 ms for its eighteen calls, against about 1.4 ms for the primitives it replaced. The Metal mask fix of Decision BO follows in its own PR.

<a id="decision-bq"></a>
## BQ: Metal's slower fused softmax

Keep both GPUs' kernels as they are (decided). With the scale and mask fused in, `Softmax [16,6,128,128]` took 0.59 ms on the M5 Pro, from 0.43, and was unchanged on the GB10, because each of its passes loads `x` and the mask again. Computing the values once, in the max pass, and storing them in the output row for the exponential pass, took Metal to 0.54 ms but the GB10 to 0.94, from 0.65: an extra write and read of 6.3 MB per layer on a GPU bound by memory. A gain of 0.05 ms on one GPU did not justify kernels that differ between them. An earlier guess, that the mask was found through `offset_of` per element, was wrong: its row's start is found once.

<a id="decision-br"></a>
## BR: after the fusions

Measure larger patches per thread in Metal's tiled matmul, then keep `T64x64` (decided). With the fusions of Decisions BM to BP, `MatMul` is 71% of a training step's GPU time on the M5 Pro and 54% on the GB10, at about 2,100–2,300 GFLOP/s forward on Metal against 4,700–5,300 on CUDA with the same kernel; memory bandwidth is not the difference, as Metal's elementwise ops are as fast as the GB10's. Each thread's 4 x 4 patch reads 8 values from threadgroup memory for 16 multiply-adds, so patches of 8 x 4 and 8 x 8, in blocks of 64 x 64, 128 x 64 and 64 x 128, were tried. All were bit-identical to `Split`, but 8 x 8 ran 2–3 times slower on every product, I think because 128 accumulators per thread cost the GPU its threads in flight, and 8 x 4 gained 3–9% on forward products, within the noise of a bench that waits for the GPU after each product. The next lever would be a different kernel, with double-buffered tiles or wider loads from device memory.

<a id="decision-bs"></a>
## BS: a first release

Tag v0.1.0 (decided): GPT-2 inference and training, on the CPU, BLAS, Metal and CUDA, with bit-identical results across their modes. M8's exit runs on a T4 are still to come.

<a id="decision-bt"></a>
## BT: an inference app

A CLI app and server inside this repo, under `apps/`, using tensomite as a library (decided). It is narrow at first and moves to its own repository once the API it uses has settled. The small instruct models on Hugging Face need BF16 weights, the Llama family's RMSNorm, RoPE, SwiGLU and grouped-query attention, and their `tokenizer.json`, none of which tensomite has yet.

<a id="decision-bu"></a>
## BU: the inference app's first scope

`apps/server`, serving GPT-2 small or a GPT-2 tensomite saved, before any Llama-family model (decided). With no arguments it is an HTTP server with OpenAI's `/v1/completions`, streamed or whole, and `/health`; `server generate PROMPT` runs the same engine once in the terminal. Requests run one at a time; text is streamed only up to whole characters and held back while it could begin a stop string. GPT-2 is a base model, so it completes text rather than chatting; a chat endpoint comes with the first instruct model.

<a id="decision-bv"></a>
## BV: training something worth serving

Fine-tune pretrained GPT-2 small on a chosen corpus, rather than train from scratch (decided). It needs a trainer for `GPT2` with BPE, the corpus tokenised once and read through a memory map, checkpoints of the model and AdamW's state with resume, and periodic validation and samples; Tiny Shakespeare in GPT-2's tokens for tests, a larger corpus overnight. First `bench/train_gpt2.cr` measures a step of GPT-2 small on each GPU; `GPT2#forward` gains a batched form without a cache for it, whose rows give each sequence's own prefill bit for bit. Measured at batch 8 × 256: 402 ms a step on the GB10, 5,089 tokens/s, and 768 ms on the M5 Pro, 2,665 tokens/s, with the same losses on both; validation loss on Tiny Shakespeare fell from 4.16 to 3.60 in 20 steps. `MatMul` is 72% and 88% of the GPU time, a third of it the head's three products with the 50,257-token vocabulary. A night is about 147M tokens on the GB10, so Tiny Shakespeare (304K tokens) would be seen about 1,200 times: the trainer needs early stopping on validation loss for small corpora, and a corpus of tens of millions of tokens for overnight runs. Metal allocates 673 buffers a step outside its pool, 22 ms, where CUDA allocates none. Then `apps/finetune`: the text tokenised once into a file of 16-bit ids read through a memory map; each step's batch drawn by a generator seeded with the seed and the step, so a checkpoint needs no generator state; checkpoints of the weights, AdamW's state (`Optimizer#state` and `#restore`) and the step, written whole and then swapped in; a warm-up and cosine learning rate; validation on fixed windows of the text's last tenth, keeping the best weights and stopping after a patience; and `best/` and `model/` directories that `apps/server` serves. A run resumed from a checkpoint matches one that never stopped, bit for bit.

<a id="decision-bw"></a>
## BW: real models and quantisation

A Llama-family instruct model at F32 first (BF16 loading, RMSNorm, RoPE, SwiGLU, grouped-query attention, `tokenizer.json`, a ChatML template), then `Q8_0` and `Q4_0` (M10), measured against that F32 baseline by perplexity; FP4 formats later, if a model needs them (decided). Tensor cores, if ever, would be a precision setting on the existing GPU devices, not a device of their own, as they give up bit-identity.

<a id="decision-by"></a>
## BY: after the first fine-tune

Time only the training steps in `finetune`'s progress, then add dropout (decided). Fine-tuning GPT-2 small on Tiny Shakespeare found its best validation loss, 3.34 from about 4.16, at step 200, after about 1.3 passes over the text; by step 700 training loss was 2.3 and validation 3.76, and patience stopped the run. Dropout is a primitive, `Op::Dropout`, whose mask is a hash of the seed and each element's position in 32-bit integer arithmetic, so every device makes the same mask and no random numbers are uploaded; its gradient is the same mask on the gradient. GPT-2 takes a `Dropout` in training and applies it where Hugging Face's does, and `finetune` uses 0.1 by default, seeded by the step, so resumes keep their bits. Measured on the same run with dropout 0.1, validated every 50 steps: the best validation loss came at step 500, 3.315, against step 200 and 3.344 without; by step 700 it had risen 0.03 from its best, against 0.41, and the gap between training and validation loss was about 0.35 rather than 1.45. Overfitting nearly stops, but the best loss barely moves, so the size of the text, 300K tokens, is now the limit.

<a id="decision-bz"></a>
## BZ: a corpus of code

Fine-tune GPT-2 small on the Crystal standard library and measure it on held-out methods (decided). `scripts/make_code_corpus.sh` concatenates the library's 1,646 files, 9.1 MB or 4.1M tokens (2.2 bytes a token, since GPT-2's vocabulary has few tokens for runs of spaces), in an order shuffled by a checksum of each path, so the validation tenth is whole files from every part of the library. `bench/code_eval.cr` takes about 40 methods from those files. Each model is shown the file up to a `def` line and measured four ways: the loss of the real body; whether its greedy completion closes with an `end` at the `def`'s indent; whether the file with that completion parses; and whether the first line, or the whole completion, matches the real one. It doesn't type-check, because Crystal checks only methods a program calls, so a type check would pass any body; tests to run would need code written for them. Measured on 40 held-out methods after 6,000 steps on the GB10 (batch 4 × 512, dropout 0.1, 3.3 passes over the text; validation loss 1.22 at step 250 and 0.99 at the end, still falling, with training loss level with it): body loss fell from 2.03 to 1.09; completions that close went from 6 to 30, that parse from 6 to 29, and first lines that match from 1 to 8; none matched exactly. GPT-2 small mostly repeats comments or writes other languages; the fine-tuned model writes Crystal's grammar but often the wrong logic, and its failures are mostly loops of greedy decoding, such as `builder.builder.builder…`. A sample prompt that ends in a space, such as `"  def "`, gives samples of only spaces: GPT-2 attaches a space to the word after it, so a lone space token is almost always indentation, and after one the next token is another space 77% of the time in this corpus.

<a id="decision-ca"></a>
## CA: the CUDA context across threads

Make the context current on whichever thread calls the driver, rather than run GPU work in an `Isolated` execution context (decided). Crystal 1.22 moves a fiber to another thread after a file write, and a CUDA context is current per thread, so a fine-tune failed with `CUDA_ERROR_INVALID_CONTEXT` after exporting its best weights. Each `CUDA::Context` call now binds the context when the calling thread differs from the last one's, a thread-local read of 3.8 ns, and the host stub keeps a context per thread so the specs catch a call that skips it. Pinning the work with `Isolated` would have made correctness depend on every caller, and finalizers, which free device memory on whatever thread collects, would still need the check.

<a id="decision-cb"></a>
## CB: a bigger model to fine-tune on code

GPT-2 large rather than XL (decided). XL's first step took 5.0 s on the GB10, but its memory ran out early in the second; large measured 2.6 s a step with about 67 GB in use. Its runs were then killed for memory too, every few hundred steps, which Decisions CC and CD address.

<a id="decision-cc"></a>
## CC: memory in the progress lines

`finetune` reports the system's memory in use, on CUDA the device memory tensomite holds (`CUDA#device_bytes`) and its pool's share, and the collector's heap (decided). The large runs were killed by the kernel with nothing in the logs to say what had grown; the lines then showed memory flat for hundreds of steps and the process killed holding about 131 GB, the state plus two steps' worth.

<a id="decision-cd"></a>
## CD: deterministic release of a step's memory

Free each intermediate buffer as soon as the last node of the same realize that reads it has run, opt-in through `Tensor.realize(tensors, release: true)`, which `Optimizer#step(loss)` uses (decided). The collector cannot see device memory, and it is conservative: a stale pointer to any of the step's collections of nodes, such as the scheduler's list or autograd's table of gradients, keeps every buffer of the step, about 60 GB on GPT-2 large, and the next step's 60 GB then exceeds the GB10. Emptying those collections would patch the holders found so far; releasing at the end of the step would be deterministic but leave the peak; releasing as the step goes is both, since the forward pass's activations go back to the pool as the backward pass finishes with them. Roots keep their buffers, as do `Write` nodes and the buffers they write into, and a released node raises when read. Reuse within the step relies on each device running work in the order it is submitted: one CUDA stream, one Metal queue. Inference still realises without it.

<a id="decision-ce"></a>
## CE: the schedule for a bigger model

End the learning-rate schedule near where validation peaks, 3,500 steps for GPT-2 large on the Crystal corpus (decided). With the 9,000-step schedule large's best validation loss, 0.885, came at step 3,250, while the learning rate was still 3.2e-5, and patience stopped it at 4,750. Ending the cosine decay at 3,500 gave 0.868, and a body loss of 0.786 on 200 held-out methods against 0.807. Large still began to memorise within the last few hundred steps, after about 1.8 passes over the text, where small did not in 3.3.

<a id="decision-cf"></a>
## CF: recording the code fine-tunes

The 200-method results, the schedule finding and an analysis of first lines, with `scripts/compare_first_lines.py` to repeat it (decided). On 200 held-out methods, fine-tuned large closes 84–87% of completions and parses 82.5–85%, against small's 80.5% and 79.5%, with a lower body loss, but matches the real first line of 14–16 methods against small's 23. Of the methods only one model matched, 11 went to small and 4 to large; at least 7 of small's 11 copy the opening of a neighbouring method in the prompt, such as `check_division_by_zero other` in `BigRational#%`, where large writes plausible code of its own. Continuing the `def` line, which can never match, does not explain it: large does so less often than small. Why the larger model copies less is not known; it saw the text 1.9 times against small's 3.3.

<a id="decision-cg"></a>
## CG: after the review of 7 November

Llama-family models at F32 become M9 and quantisation M10, renumbered rather than only reordered, as Decision BW decided; the apps get a section of their own; Decision AA is recorded as done for training (decided). References to M9 and M10 in earlier decisions were renumbered with them. Before M9: a GPT-2 XL fine-tune on the GB10, then Metal's allocations that miss its pool (673 a step) and a fused `LogSoftmaxGrad` (about 13 ms a step on the GB10, the same for any GPT-2 size, so it does not hold up the XL run). M8's T4 exit stays open until a T4 is at hand.

<a id="decision-ch"></a>
## CH: after the XL fine-tune

Record it, then start M9 (decided). GPT-2 XL on the Crystal corpus, 3,000 steps at 5.1 s each with about 65 GB in use throughout, reached a best validation loss of 0.855 at step 2,750, against large's 0.868; on 200 held-out methods its body loss is 0.764 against large's 0.786, with 165 that parse against 165, 20 first lines against 16 and 5 exact against 2. Doubling the model again bought a fifth of what going from small to large did (0.148 then 0.022), so the 3.7M tokens of text, not the model, now set the limit.

<a id="decision-ci"></a>
## CI: coalesced row ops on CUDA

In the pass of a fused row op that writes each element, thread `t` takes elements `t`, `t + threads` and so on, rather than its own chunks of 16 (decided). A warp's 32 threads read addresses 64 bytes apart in the chunked order, which the GB10 served at about 91 GB/s for `LogSoftmaxGrad` where a plain `Sub` over the same rows ran at about 241; the Apple GPU's caches absorb it, at 213 GB/s. Each element's result depends only on its inputs and the row's totals, so the order changes no bits; the sums keep their chunked order. Measured on the GB10 for GPT-2 small: the six row ops went from 57.2 to 44.3 ms a step, and the step from 401.3 to 388.4 ms. `LayerNormGrad` fell from 8.0 to 4.6, `Softmax` from 8.0 to 4.8, `LogSoftmaxGrad` from 13.1 to 10.8, `LogSoftmax` from 16.7 to 14.6 and `LayerNorm` from 4.9 to 3.0; `SoftmaxGrad`, whose rows of 256 run 16 threads a block, did not move. The cross-entropy pair still moves about 140–150 GB/s against about 240 for a plain elementwise op, which is the sums' chunked reads; coalescing those, through shared memory to keep their order, is left as a follow-up.

<a id="decision-cj"></a>
## CJ: M9 in five PRs

`BF16` loading; Llama's pieces (SiLU, RMSNorm, RoPE, grouped-query attention) as composites of existing ops, fused only where a profile shows the need; `tokenizer.json`; the Llama model with fixtures from PyTorch; then ChatML and a chat endpoint (decided). `BF16` weights widen to `F32` in `SafeTensors#[]`, so every device and op sees `F32` as before; SmolLM2-135M-Instruct's 134.5M weights widen in half a second.

<a id="decision-ck"></a>
## CK: after M9

Tag v0.2.0, then M10 (decided). v0.2.0 holds everything since v0.1.0: fused training, `apps/server` and `apps/finetune`, fine-tuning on code, a training step's buffers released as it runs (Decision CD), and M9, Llama-family models at F32 with chat. A chat without a system message gets the one its chat template adds, as Hugging Face's own template does (Q-BQ, decided), so SmolLM2 answers as it was tuned to; `--system` overrides it.

<a id="decision-cl"></a>
## CL: M10 in five PRs

The GGUF reader with `F16`, the tokeniser from GGUF metadata and `Llama` from GGUF; block dtypes Q8_0, Q4_0 and Q4_1 for leaves stored `[outputs, inputs]`, from GGUF and from `Tensor#quantize`; the CPU oracle kernels, matmul with activations quantised to Q8 and integer dot products as ggml does, and a perplexity bench; Metal kernels; CUDA kernels (decided). Q4_1 joins the plan because the published Q4_0 SmolLM2 holds 3 Q4_1 tensors and a Q8_0 embedding. Quantised weights keep GGUF's `[outputs, inputs]`, with blocks along the inputs, because each output is then a dot product along whole blocks. llama.cpp's converter reorders query and key rows for its rotary layout, and loading reorders them back.

<a id="decision-cm"></a>
## CM: measuring quantisation

M10's exit is judged by the mean KL divergence from the F32 model, over 16 windows of 128 tokens of Tiny Shakespeare, not by perplexity (decided). On a text a model is too sure of, quantisation noise lowers perplexity: GPT-2 "improves" from 139.4 to 126.6 with its matrices at Q4_0, while its KL divergence is 0.080. Two rules:
  1. A format's budget, as a guard rail: at most 0.01 nats a token at Q8_0, and 0.3 at Q4_0, with embeddings at Q8_0 or better. Measured on the CPU: SmolLM2 0.005 (Q8_0), 0.154 (llama.cpp's Q4_0) and 0.252 (tensomite's plain Q4_0); GPT-2 0.001 (Q8_0) and 0.080 (Q4_0), with embeddings at F32, since its tied embeddings at Q8_0 add about 0.05. Small models suffer most from 4 bits, and better quantisers, such as llama.cpp's importance matrix, are a later project.
  2. A kernel's budget: a GPU's predictions stay within 0.001 nats a token of the CPU's on the same quantised model, so GPU kernels add next to nothing to what quantisation costs.

<a id="decision-cn"></a>
## CN: quantised kernels correct first

M10's GPU kernels take one thread per output, summing its blocks in the host's order, and a follow-up makes decode fast on both GPUs (decided). On the M5 Pro, Q4_0 decodes only 1.5 times faster than F32, though it reads 7 times fewer bytes: about 9 GB/s of weights, against 41 at F32. Too few threads share each output, and every token pays for hundreds of dispatches. The follow-up gives each output a SIMD group of 32 threads that share its blocks and add their sums at the end, which changes the order of additions, as Decision CM's kernel budget allows.

<a id="decision-co"></a>
## CO: after M10

First the decode follow-up of Decision CN, then M11 (decided). A first try gave each block's 32 elements to 32 lanes and added their integer products across the warp: exact, but five shuffles per lane per block made decode on the GB10 slower, 44 tokens a second against 93. The second gives each lane whole blocks instead, lane l taking blocks l, l + 32 and so on, and adds the lanes' contributions in block order, so it keeps the host's bits and does no more integer work than one thread per output. `BlockMatMulMode` picks lanes for products of fewer than 65,536 outputs and threads otherwise; `TENSOMITE_BLOCK_MATMUL` sets the mode, so the modes can be timed against each other. On the GB10, SmolLM2's Q4_0 decodes at 88.5 tokens a second on threads and 169.4 on lanes; with a threshold of 8,192 outputs, which gave its 49,152-word head threads, at 163.8, so the threshold rose to 65,536, and every product of one-token decode runs on lanes. On 16 windows of 128 tokens, where most products are wide, the modes are within noise of each other: 1.3 s on threads, 1.4 on lanes and 1.2 on `Auto`. On the M5 Pro, the modes make no difference: about 120 tokens a second in decode and 1.2 s for the 16 windows, in each mode, so its decode is bound by something other than the quantised matmul.

<a id="decision-cp"></a>
## CP: profiling before M11

The profilers take Llama-family models and GGUF files as well as GPT-2, as `bench/profile_decode.cr` (Metal) and `bench/profile_decode_cuda.cr` (CUDA), before any work on M11 (decided). On the M5 Pro, quantised decode runs at the same 120 tokens a second whether its matmul runs on threads or lanes, so something else bounds it there; the profile, which splits a decoded token into encoding, allocation, waiting for the GPU and host ops, and GPU time by op, says what.

<a id="decision-cq"></a>
## CQ: fusing Llama's composites

RMSNorm becomes one row op and RoPE one elementwise op, before M11 (decided). The profiles of SmolLM2's Q4_0 decode show 1,450 dispatches a token on the M5 Pro and 1,480 launches on the GB10, each costing about 3 µs however little it does: 4.3 ms of a 6.0 ms token waiting for the GPU on the M5 Pro, 2.9 ms of 5.2 launching on the GB10. RMSNorm's composite was 7 ops and RoPE's 5, run 61 and 60 times a token. Fused, the tiny Llama's decode goes from 87 ops a token to 41. The fused ops compute what the composites compute, operation for operation, on host devices, so the host's logits do not change; RoPE, which only multiplies, negates and adds, gives the same bits on GPUs, and RMSNorm sums its rows in chunks there, as LayerNorm does. Both keep gradients: RMSNorm's as a composite, and RoPE's as `g · cos` plus the transpose of `rotate_half` applied to `g · sin`. Measured after it, per decoded token of SmolLM2's Q4_0: the M5 Pro went from 6.03 ms and 1,450 dispatches to 4.72 ms and 784, and the GB10 from 5.21 ms and 1,480 launches to 3.61 ms and 814, with its uploads down from 154 to 2, since the composites had uploaded their constants each token.

<a id="decision-cr"></a>
## CR: joining projections that share an input

`Linear.merge` joins the query, key and value projections, and the MLP's gate and up, so a Llama block runs four products a token instead of seven (decided). After Decision CQ, the quantised products were the largest cost left, 58% of the M5 Pro's GPU time, at two dispatches each. Quantised weights join by rows and `F32` ones by columns; each part's weight becomes a view of its share, so nothing is held twice, and `Llama` joins them when it loads or quantises. A projection whose weight is replaced, as in training, runs on its own again. Each output is still its own product of the same quantised input, so the logits keep their bits: the tiny Llama's decode goes from 41 ops a token to 35, and SmolLM2's should lose 90 products, 180 dispatches.

<a id="decision-cs"></a>
## CS: a larger model before choosing

Profile SmolLM2-1.7B at Q4_0 before choosing decode's next work, since the 135M model mostly measures dispatch overhead (decided). After Decision CR, the 135M model decoded in 3.76 ms a token on the M5 Pro and 3.15 on the GB10, from 4.72 and 3.61. The published 1.7B Q4_0 GGUF stores its embeddings as Q6_K, which tensomite cannot read, so the profilers take `--quantize TYPE` to quantise the original weights after loading. The 1.7B model took 7.00 ms a token on the M5 Pro and 7.11 on the GB10, against 13 times the weights, with quantised products 71–73% of GPU time: its decode is bound by reading weights.

<a id="decision-ct"></a>
## CT: writing quantised models

Use llama.cpp's `llama-quantize --pure` to make GGUF files for now, rather than a writer of tensomite's own (decided). Its Q4_0 matches tensomite's quantiser byte for byte, and `--pure` keeps every matrix at the one format; without it, Q4_0 stores embeddings as Q6_K. Revisit when tensomite makes formats llama.cpp does not.

<a id="decision-cu"></a>
## CU: the GPUs before M11

Close the gap to llama.cpp's GPU decode before M11's CPU kernels (decided). On SmolLM2-1.7B at Q4_0, llama.cpp decodes in 4.73 ms a token on the M5 Pro and 4.85 on the GB10, 1.5 times faster than tensomite on both; on the GB10's CPU it takes 11.4 ms, 88 tokens a second, M11's target. About 963 MB of weights a token at the GB10's 273 GB/s puts the floor near 3.5 ms. The work, in order: RMSNorm, the quantised products, then attention.

<a id="decision-cv"></a>
## CV: RMSNorm's order

A fast GPU RMSNorm may sum its row in its own order (decided). The GPUs already summed in chunks, within 1e-5 of the host, so no bits were given up. Up to 512 threads a row, reading neighbouring elements together and adding in a tree, took a row of 2,048 from 10.3 µs to 6.2 on the M5 Pro and from 12.5 to 6.9 on the GB10, 0.2 ms a token; an `Add` of the same row takes 2.8.

<a id="decision-cw"></a>
## CW: the quantised products

Wider reads and more outputs a group, keeping the host's bits (decided). On CUDA, weights read two bytes at a time and activations four, multiplied four pairs an instruction with `__dp4a`, took SmolLM2-1.7B's products from 5.68 ms a token to 4.71, the whole token to 5.9: one output a warp was fastest, and two, four or eight were no faster, though grouped outputs share their activations. Adding in block order costs CUDA nothing: the same kernel adding in a tree, `BlockMatMulMode::Unordered`, took 4.88. On Metal, which has no `__dp4a`, the same design was slower than the byte-at-a-time kernel at every count of outputs a group: emulating `__dp4a` with shifts gave 8.78 ms at one output and 5.97 at four, against 5.14; summing Q4 products in floats, exactly, since each is an integer below 2^24, gave 5.29 at four. Grouping helped Metal, by a third from one output to four, and adding in block order cost it 0.36 ms (5.29 against 4.93 unordered). Shared memory reserved for 32 outputs a group, 17 KB, slowed both GPUs by leaving room for fewer groups.

<a id="decision-cx"></a>
## CX: Metal's lanes kernel

Metal keeps the byte-at-a-time kernel, one output a SIMD group, and `Unordered` stays on both GPUs for measurement (decided). Neither exactness nor grouping explains the gap to llama.cpp on the M5 Pro: unordered, the products still took 4.93 ms. llama.cpp's Metal kernel differs in what remains: half a block a lane, 16 blocks a pass, and `F32` activations with no quantising pass. A next attempt starts from those, measured against 5.14 ms.

<a id="decision-cy"></a>
## CY: after the quantised products

Fused attention next, before reworking Metal's lanes kernel or starting M11 (decided). Decoding SmolLM2-1.7B, attention is three dispatches a layer, a matmul for the scores, a softmax and a matmul with the values, 72 a token and 0.7–0.9 ms of GPU time at about 80 positions, writing the scores and weights out and reading them back; it grows with the context.

<a id="decision-cz"></a>
## CZ: how attention fuses

`Tensor.attention` makes one `Op::Attention` on GPUs for at most 8 query rows per head and 4,096 positions, as decoding has, and the composite otherwise; host devices always compute the composite, so their bits do not change (decided). The kernel takes one threadgroup per query row of a head: scores into shared memory, their maximum and sum, then the weighted values. It adds in the orders the composite's kernels do: dot products in chunks of 64, as the matmul kernels sum, the exponentials in chunks of 16, as softmax does, and the weighted values in chunks of 64. So on CUDA it has the composite's bits, and decoding one token at a time still gives a prefill's logits bit for bit, as shape invariance (Decision K) requires. Its gradient recomputes the softmax weights from the composite. A first kernel found every key and value with `offset_of`'s divisions and left each output column's sum to one thread: 1.73 ms a token on the GB10 and 1.33 on the M5 Pro, against 0.73 and 0.92 for the three ops. Stepping through each head's keys and values by stride, and sharing each column's sum among threads, gave 0.42 ms and 0.70: SmolLM2-1.7B at Q4_0 then decodes in 5.71 ms a token on the GB10 and 6.60 on the M5 Pro.

<a id="decision-da"></a>
## DA: after fused attention

Rework Metal's quantised products along llama.cpp's lines, then cut CUDA's launch overhead (decided, then revised by Decisions DB and DC). After Decision CZ, SmolLM2-1.7B at Q4_0 decoded in 6.60 ms a token on the M5 Pro, 1.40 times llama.cpp's 4.73, and 5.71 on the GB10, 1.18 times its 4.85; the M5 Pro's quantised products took about 5.1 ms of the 6.6.

<a id="decision-db"></a>
## DB: quantising inside the product

Activations stay quantised by their own pass (decided). Quantising a decoded row inside each product's kernel, into each threadgroup's shared memory, removed a dispatch per product, 97 a token, but every threadgroup of 4 outputs quantised the whole row again before its first product: the products took 10.13 ms a token on the GB10, from 4.7–4.9, and 8.86 on the M5 Pro, from 5.16. Folding one pass into another pays only when the work duplicated is small beside the dispatch saved.

<a id="decision-dc"></a>
## DC: CUDA graphs

Deferred, and M11 next (decided). Decoding SmolLM2-1.7B on the GB10, the CPU spends about 1.1 ms of a 5.7 ms token launching kernels and then waits about 4.3 ms for the GPU: the launches overlap the GPU's work, which bounds the token, so graphs, which cut the CPU's cost of launching, would save little more than the GPU's gaps between kernels. They would matter for small models, where launching and waiting are about equal, as for the 135M model's 1.2 ms and 1.4; a decoded token's changing arguments, such as the cache's length, would also need each replay's parameters updated. Revisit when small models matter, or when the GPU's own time falls near the launches'.

<a id="decision-dd"></a>
## DD: M11 in four parts

A CPU decode profiler; a fast single-thread quantised matmul with the oracle's bits; its outputs shared among threads; then the host kernels the profile shows next (decided). The fast kernels serve both host devices, `CPU` and `BLAS`, since they give the oracle's bits, and the naive version stays as `Quant.matmul_reference`, which the specs compare them with (Decision M's oracle). On SmolLM2-1.7B at Q4_0, decoding on the CPU went from 414 ms a token to 108 on one thread on the GB10, and from 365 to 88 on the M5 Pro; on all threads, to 19.5 and 14.0, against llama.cpp's 11.4 and 6.7. A first pool, a `Fiber::ExecutionContext::Parallel`, gave no gain on the M5 Pro and a loss on the GB10: a product's spans take 50–200 µs, and its parked threads woke after the first had done them all. Workers that spin on a counter, one fewer than the threads with the calling thread working too, take a span within a microsecond.

<a id="decision-de"></a>
## DE: building for the CPU at hand

Builds stay portable by default, and `ops.yml` offers native ones, `ops bn` and `ops run-native` with `--mcpu=native`, leaving the choice to whoever builds (decided). Native builds decode SmolLM2-1.7B 11% faster on the GB10 and 17% on the M5 Pro on all threads, and 1.6–1.85 times faster on one; enabling the 8-bit dot product alone (`--mattr=+dotprod`) changed nothing. That flag tested whether LLVM would use `sdot`, not what it is worth: the Q4 loop then multiplied unsigned nibbles by signed bytes, which only `usdot` (the i8mm extension, which `native` enables on both machines) matches, so `+dotprod` builds kept widening multiplies (amended by Decision DF). Hand-written `sdot` in inline assembly was set aside: code to maintain, on one architecture.

<a id="decision-df"></a>
## DF: after reviewing issues #113–#127

Fix the pool (#113, #125), take the serial work out of each quantised product, then rework the Q4 loop, keeping the oracle's bits, before `Tensor#to_a` (#123) and the documents (decided). On `main`, decoding SmolLM2-135M on the CPU hung in `SpinPool#run` here: a worker's stale span claim, as #113 described. Before a product's threads started, copying its activations through `offset_of` and quantising them on one thread took 2.1 ms of a 16.8 ms token (135M, 4 threads); reading dense activations in place, and quantising through pointers with rounding by truncation, took 0.23 ms. The Q4 loop then subtracts 8 from each nibble as it unpacks, so both sides are signed and LLVM emits `sdot` under `+dotprod` or `native`; the integer is the same, since Σ(q − 8)·y = Σq·y − 8·Σy. Two weight rows share each pass over the activations. `F16` scales stay a lookup in a table of every half's value: LLVM's `llvm.convert.from.fp16` failed to link with Homebrew's Crystal 1.22.2, inline assembly was set aside, and widening in registers (the bits moved to a `Float32`'s place and multiplied by 2¹¹², exact for every half) cost about 1 ns more a scale than the lookup, one scale for every 32 weights, which slowed SmolLM2-1.7B's tokens by 3–5 ms on both machines. With the three changes, SmolLM2-1.7B at Q4_0 decodes on all threads in 7.17 ms a token on the M5 Pro (6.36 native) and 12.8 on the GB10 (11.2 native), from 10.7 and 16.6, against llama.cpp's 6.7 and 11.4; the quantised products are 84–89% of a token. Each output still adds its blocks in order, so the bits are the reference's.

<a id="decision-dg"></a>
## DG: after M11

M11 closes, then the documents (#114 to #120: first the stale parts of `DESIGN.md`, `HANDOFF.md`, `README.md` and `DEVELOPMENT.md` and `Quant`'s comments, then a glossary, then the measurement history and the decisions moved into files of their own), then generation and the server (#121, #122, #127) (decided). The remaining CPU costs are small beside the products, 84–89% of a token, and the GPUs' gap to llama.cpp, 1.40 times on the M5 Pro, is now the project's largest.

<a id="decision-dh"></a>
## DH: the understanding checks since Q-BO

Written up in `faq/` now, before the glossary, which links to them (decided). Sixteen pages: Q-BP, Q-BR, Q-BT to Q-BY, Q-CA to Q-CC, Q-CE to Q-CG, and two checks from M11 that had no label, Q-CI (a table against arithmetic) and Q-CJ (the serial share). Checks never answered have their answers folded, as every page does, so each can still be tried first. Q-BQ, Q-BS and Q-CD were decisions, not checks.

<a id="decision-di"></a>
## DI: what the design keeps of its history

`DESIGN.md` describes the system as it is, with one table of current baselines, so that a regression shows against a number; a lesson learned the hard way stays as a short `> [!NOTE]` "Why this design?" callout that names its decision, so it is not relearned; the dated before-and-after figures go, since the decisions and the pull requests hold them (decided). This revises #119: the decisions move into a file of their own, and no measurement notebook is kept.

<a id="decision-dj"></a>
## DJ: the order of generation and the server's issues

One `Cache` and one `generate` for both models first (#127), then the head on the last position only (#121), then sampling without sorting the vocabulary (#122) (decided). The shared `generate` meant the other two changed one copy rather than two. With them, a 64-token GPT-2 prefill that keeps only the last row ran at 712 tokens/s against 578 on `BLAS` in the Linux container, and a top-40 pick from 50,257 logits took 197 µs rather than 6,603, choosing the same tokens for the same seed.

<a id="decision-dk"></a>
## DK: delegating to a cheaper model

Well-specified, mechanical tasks go to a subagent on Sonnet: refactors whose checks prove them, benches, specs, documentation tables (decided). Design, diagnosis and review stay with the main session, which writes each task's brief, reviews the diff and reruns the checks; work whose difficulty is judgement, such as a race, a regression or what a callout should say, is not delegated. Each pull request says what was delegated. Reviewed after the first delegated task, to see whether it saved more than the brief and the review cost.

<a id="decision-dl"></a>
## DL: the subagent's standing brief

An agent definition, `.claude/agents/implementer.md`, holds what every delegated task needs: where the conventions are, the rules that matter (bits, hot loops, comments, documents), the checks and their commands, what not to do (commit, push, open pull requests), and how to report (decided). A task's brief then carries only the task. Its last section, lessons from earlier tasks, grows as reviews find mistakes worth not repeating. Sonnet only, until the approach is shown to help.
