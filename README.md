# tensomite.cr

A small pure-Crystal tensor maths runtime with a minimal native seam to BLAS and the GPU.

Tensomite gives Crystal lazy tensors with automatic gradients, for training models and running them. Its kernels are written in Crystal: they run on the CPU, through BLAS (Accelerate or OpenBLAS), or on an Apple GPU through Metal, and repeat their results exactly from run to run. It loads GPT-2 from Hugging Face's safetensors files.

```crystal
require "tensomite"
include Tensomite

x = Tensor.from([1, 2, 3].map(&.to_f32), [3])
loss = (x * x).sum
Tensomite.grad(loss, [x]).first.to_a(Float32)   # => [2.0, 4.0, 6.0]
```

## Using it

- [User's guide](GUIDE.md): the API, with small examples.
- [Tutorial](TUTORIAL.md): train a tiny network to multiply, then use it.
- `examples/shakespeare.cr` trains a character-level GPT; `examples/gpt2_diagram.cr` draws GPT-2's structure.

## Working on it

- [`PLAN.md`](PLAN.md): goals, milestones and decisions.
- [`DESIGN.md`](DESIGN.md): how it works, and its measured performance.
- [`DEVELOPMENT.md`](DEVELOPMENT.md): setup, commands, conventions and workflow.
- [`HANDOFF.md`](HANDOFF.md): where things stand, for the next session.

Licensed under the MPL-2.0.
