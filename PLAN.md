# Tensomite: plan

Oct 22, 2026

## Goal and principles

The aim is a minimal Crystal shard for tensor maths, inference and training on macOS and Linux, written almost entirely in Crystal. It must be fast by design, not fast later by rewrite. So the hard parts of a production runtime (lazy execution, device-owned memory, runtime dtypes) come first, in small steps.

- **Lazy by default.** Ops build a graph and `realize` executes it. This gives one place to schedule, fuse, plan memory and differentiate.
- **Memory belongs to devices.** A tensor never assumes it can read its own bytes. Unified memory (Apple Silicon) is an optimisation of the discrete case, not the default.
- **Narrow FFI waist.** Native calls go through two seams only: CPU BLAS and a GPU driver. Adding an op never adds an FFI binding.
- **Kernels live in Crystal.** GPU kernel source is written or generated in Crystal. MSL is compiled by the Metal driver at runtime, and GLSL is compiled to SPIR-V at build time.
- **CPU backend is the oracle.** Every other backend is tested against the naive pure-Crystal implementation.
- **Always runnable.** Each milestone ends with something that runs end to end and has a measurable exit.

## Architecture

```mermaid
flowchart TD
    App["Models, training loops, sampling"] --> NN["NN layer: modules, optimisers, loaders, tokeniser"]
    NN --> Lazy["Tensor (lazy): shape, dtype, device, graph node"]
    Lazy --> Passes["Graph passes: autograd, fusion, memory planning"]
    Passes --> Sched["Scheduler: topo order, kernel selection, transfers"]
    Sched --> Iface{"Device interface: buffers, transfer, run kernel, sync"}
    Iface --> CPU["CPU: pure Crystal (oracle)"]
    Iface --> Sim["SimDiscrete: CPU with forced copies (test only)"]
    Iface --> BLAS["BLAS: matmul only"]
    Iface --> Metal["Metal: MSL strings"]
    Iface --> Vk["Vulkan: GLSL strings"]
    BLAS -. "FFI: cblas_sgemm" .-> LibBLAS["Accelerate / OpenBLAS"]
    Metal -. "FFI: ~17 calls via objc_msgSend" .-> MetalFw["Metal.framework"]
    Vk -. "FFI: ~40 calls + structs" .-> VkLib["libvulkan"]
    Vk --> Spv["SPIR-V compiled at build time (glslc)"]
    style LibBLAS stroke:#d9480f,stroke-width:3px
    style MetalFw stroke:#d9480f,stroke-width:3px
    style VkLib stroke:#d9480f,stroke-width:3px
    style CPU stroke:#2b8a3e,stroke-width:3px
```

### Core types

- **`DType`:** a runtime enum (`F32`, `F16`, `BF16`, `I32`, `U8`, then block formats `Q8_0`, `Q4_0`). Block formats have a block size and bytes per block rather than an element size.
- **`Buffer`:** an opaque, device-owned allocation with a byte size. Only its device can read it. `Buffer#map` returns a host `Bytes` view when the device has unified memory, and `nil` otherwise.
- **`View`:** shape, strides and offset. It is pure metadata, so reshape, transpose, slice and broadcast never touch data.
- **`Tensor`:** non-generic. It holds a `View`, a `DType`, a `Device` and either a realised `Buffer` or an unrealised graph node. Typed host access is explicit: `t.to_a(Float32)`, `t.to_slice(Float32)`.
- **Ownership:** a `Buffer` is released by its device, via a finaliser and an explicit `release`. Views keep their buffer alive. mmap'd files are a `Buffer` on the CPU device.

### Execution flow

```mermaid
flowchart LR
    Op["a.matmul(b) + c"] --> Node["Graph nodes (no work done)"]
    Node --> Realize["realize / to_a"]
    Realize --> Topo["Topological order"]
    Topo --> Fuse["Fusion pass (from M6)"]
    Fuse --> Plan["Allocate / reuse buffers"]
    Plan --> Xfer["Insert transfers where devices differ"]
    Xfer --> Run["Dispatch kernels"]
    Run --> Sync["Sync, return buffer"]
```

Before M6 the fusion pass is the identity, so each node becomes one kernel.

### Memory model

Discrete GPUs (CUDA, or Vulkan on a graphics card) have their own memory across a PCIe bus. Unified GPUs (Apple Silicon, integrated Intel/AMD) share RAM with the CPU. The device interface always speaks in transfers. A unified device implements a transfer as a no-op, or as a `map`.

The `SimDiscrete` device runs CPU kernels on private buffers and refuses host access except through explicit transfers. It finds residency bugs on any laptop, long before a Vulkan GPU is involved.

## Milestones

Each milestone is done when its exit criterion passes.

1. **Skeleton.** Build `DType`, `View`, `Buffer`, the device interface, the CPU device, lazy `Tensor`, and a `realize` that interprets the graph node by node. Implement elementwise add/mul and `sum`.
   - Exit: `(a + b).sum.to_a(Float32)` matches by hand, including over transposed and broadcast views.
2. **CPU op set and SimDiscrete.**
   - Elementwise and unary ops: exp, log, tanh, GELU (tanh approximation).
   - Reductions: sum, max, mean over an axis.
   - Composite ops: softmax, layernorm, embedding gather, causal mask, slice/concat, and naive matmul.
   - Exit: specs pass against hand-computed values, and the same suite passes unchanged on `SimDiscrete`.
3. **BLAS seam.** Bind `cblas_sgemm` for Accelerate and OpenBLAS. Non-contiguous inputs are made contiguous first.
   - Exit: results are within a relative error of 1e-5 of the naive matmul, and the speedup is benchmarked.
4. **GPT-2 inference.** Write a safetensors loader (mmap as a CPU `Buffer`), a byte-level BPE tokeniser, and the GPT-2 small forward pass with a KV cache and sampling.
   - Fixtures (logits and tokens) are generated once from a PyTorch reference and committed, so CI needs no Python.
   - Exit: logits are within tolerance of the fixture, greedy decoding matches for 64 tokens, the tokeniser matches the reference on a Unicode corpus, and baseline tokens/s is recorded.
5. **Autograd and training.** Implement reverse mode as a graph transform: `grad(loss, params)` returns new lazy tensors. Add SGD, AdamW and cross-entropy.
   - Exit: gradients pass finite-difference checks, and a char-level Shakespeare model reaches a validation loss below 1.6 nats/char.
6. **Metal backend.** Bind the driver through `objc_msgSend` (one declaration per selector signature) with autorelease pools around dispatch. Add a pipeline cache, then hand-written MSL kernels, ported one op at a time.
   - Exit: every op matches the CPU oracle, and GPT-2 tokens/s beats BLAS.
7. **Fusion and memory planning.** Elementwise and reduction chains are fused into one generated MSL kernel per group, and buffers are reused across the graph. Matmul and attention stay hand-written.
   - Exit: dispatch count per GPT-2 token falls measurably, tokens/s rises, and fused results match the oracle.
8. **Vulkan backend.** Bind Vulkan compute, and port the hand-written kernels to GLSL compiled at build time. Then settle Decision I.
   - Exit: the full op suite matches the oracle on a Linux discrete GPU, and GPT-2 runs there.
9. **Quantisation.** Add block dtypes, a GGUF loader, and Q8_0/Q4_0 dequant-matmul kernels for CPU, Metal and Vulkan.
   - Exit: a quantised GPT-2 stays within a perplexity budget of f32.
10. **Llama-family model.** Add RoPE, RMSNorm, SwiGLU and grouped-query attention.
    - Exit: TinyLlama runs from GGUF on macOS and Linux, and matches reference fixtures.

## Decisions

- **Decision A, own backends or bind ggml:** own backends (decided).
- **Decision B, Linux GPU:** Vulkan (decided). It covers AMD, Intel and NVIDIA. CUDA can follow later through the same device interface.
- **Decision C, first model:** GPT-2 small, then TinyLlama (decided).
- **Decision D, CPU parallelism:** `-Dpreview_mt` or execution contexts. Defer until profiling shows the CPU path matters.
- **Decision E, execution model:** lazy graph with `realize` (proposed, confirm).
- **Decision F, tensor typing:** non-generic `Tensor` with a runtime `DType` (proposed, confirm).
- **Decision G, Metal plus Vulkan, or Vulkan everywhere via MoltenVK:** native Metal (proposed). Revisit after M8, when keeping two codegen emitters has a known cost.
- **Decision H, Vulkan shader compilation:** compile at build time (decided). A Crystal macro runs `glslc` during `crystal build` and embeds the SPIR-V, so users need no compiler and there is no shaderc seam. Fused kernels on Vulkan are deferred to Decision I.
- **Decision I, fused kernels on Vulkan:** decide at the end of M8, using the fusion gain measured on Metal in M7. The options are no fusion on Vulkan, hand-written fused kernels for common patterns (compiled at build time), shaderc at runtime for fused kernels only, or a small SPIR-V emitter in Crystal.

## Risks

- **Lazy debugging.** Shape and dtype errors are raised as soon as a node is built. Only execution errors (NaN, out of memory, kernel failures) surface at `realize`, far from their cause. Mitigate by recording the call site on each node and adding an eager debug mode that realises after every op.
- **No portable SIMD in Crystal.** CPU speed relies on LLVM auto-vectorisation under `--release`, which is weak for quantised dot products. The fallback is a small C file of SIMD kernels. That would be a fourth seam, so it needs a benchmark to justify it.
- **`objc_msgSend` signatures.** A mistyped selector signature crashes without a useful error. Mitigate by binding each selector once and covering each with a spec.
- **Vulkan surface.** Vulkan compute needs far more ceremony than Metal (descriptor sets, pipeline layouts, fences, memory types). Keep it behind one file per concern and port only after the Metal path has settled the device interface.
- **Kernel drift.** Hand-written and generated kernels can diverge from the oracle silently. Mitigate by running every op on every available device against the CPU in specs.
- **GPU CI.** It is unverified whether hosted macOS runners give usable Metal compute, and hosted Linux runners have no GPU. CI runs CPU and `SimDiscrete` on every push. GPU specs run locally, or on a self-hosted runner if one appears.
- **Scope creep.** Multi-GPU, CUDA and new architectures stay out until M10 passes.
