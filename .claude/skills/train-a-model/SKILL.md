---
name: train-a-model
description: Use when asked to define, train, save or reload a neural network in Crystal with the tensomite shard (a Module built with a Builder, a loss, AdamW, a training loop on CPU, BLAS, Metal or CUDA). Not for fine-tuning a pretrained GPT-2; use finetune-a-model for that.
---

# Train a model with tensomite

Tensomite is a Crystal tensor library with lazy tensors, automatic gradients, optimizers, and models built from modules. In a dependent project the shard is installed in `lib/tensomite/`; in its own checkout the same files are at the root. Paths below are relative to the checkout; prefix them with `lib/tensomite/` in a dependent project.

## Steps

1. **Add the shard** to the project's `shard.yml`, then run `shards install`. Tensomite needs Crystal 1.22 or later.

   ```yaml
   dependencies:
     tensomite:
       github: nogginly/tensomite.cr
   ```

2. **Require it.** `include Tensomite` lets the code write `Tensor` for `Tensomite::Tensor`.

   ```crystal
   require "tensomite"
   include Tensomite
   ```

3. **Pick a device.** The CPU is the default and needs nothing. The faster devices are opt-in requires, and each has a shared instance to pass as `device:`:

   Device |Require                                         |Instance                                                      
   -------|------------------------------------------------|--------------------------------------------------------------
   `CPU`  |none                                            |`CPU.default`: correct, slow for large matrices               
   `BLAS` |`require "tensomite/blas"`                      |`BLAS.default`: the CPU with BLAS matrix products             
   `Metal`|`require "tensomite/metal"`, on Apple Silicon   |`Metal.default`; `Metal.available?` says whether there is a GPU
   `CUDA` |`require "tensomite/cuda"`, built with `-Dcuda` |`CUDA.default`; `CUDA.available?` says whether there is a GPU 

   Give the device to the `Builder` (`Builder.new(rng: Random.new(1), device: BLAS.default)`) and to every tensor made from data (`Tensor.from(data, shape, device)`). All operands of an operation must be on one device. `examples/shakespeare.cr` picks a device from its `--device` option.

4. **Define the model** as a `Tensomite::Module` whose instance variables hold its parts. Parameters are found and named from the variables; nothing is listed by hand. Type the builder argument (`build : Builder`), or Crystal cannot infer the variables' types.

   ```crystal
   class Classifier < Tensomite::Module
     def initialize(build : Builder)
       @hidden = build.linear(2, 32)
       @output = build.linear(32, 2)
     end

     def forward(x : Tensor) : Tensor
       @output.call(@hidden.call(x).gelu)
     end
   end

   model = Classifier.new(Builder.new(rng: Random.new(1)))
   ```

   The builder makes `linear`, `embedding`, `layernorm`, `rmsnorm`, `attention`, `swiglu` and `grouped_attention`; `Sequential.build(build) { ... }` chains layers. Repeated parts go in an `Array` whose type is declared (`@blocks : Array(Block)`). Run the model with `model.call(x)`, which runs `forward`. `examples/shakespeare.cr` is a complete transformer written this way.

5. **Make batches as tensors.** Data is flat and row-major. Floats are `Float32` (dtype `F32`); class and token ids are `Int32` (`I32`):

   ```crystal
   inputs  = Tensor.from(floats, [batch, 2])   # floats : Array(Float32)
   classes = Tensor.from(ids, [batch])         # ids : Array(Int32)
   ```

6. **Train.** Give the optimizer the model's parameters once. `step(loss)` computes the gradients, reads the loss, updates every parameter and returns the loss's value. Pass it the lazy loss; do not read the loss first, because reading it frees the graph the gradients need.

   ```crystal
   optimizer = AdamW.new(model.params, lr: 5e-3, weight_decay: 0.0)
   1000.times do |step|
     inputs, classes = next_batch
     loss = optimizer.step(model.call(inputs).cross_entropy(classes))
     puts "step #{step}: loss #{loss.round(4)}" if step % 100 == 0
     GC.collect # returns the step's device buffers to the pool
   end
   ```

   `cross_entropy(targets)` takes `[n, classes]` scores and `[n]` `I32` class ids. For regression, use `error = model.call(x) - y` and `(error * error).mean`. `SGD.new(params, lr:, momentum:)` is the simpler optimizer. For a learning-rate schedule, assign `optimizer.lr = rate` on an `AdamW` before each step.

7. **Save and load.** `model.save("model.safetensors")` writes every parameter under its name. To load, build a model of the same shape and call `Classifier.new(builder).load("model.safetensors")`. A missing or extra name raises `Tensomite::Error`, and a different shape raises `ShapeError`. The file is standard safetensors. To resume an optimizer too, write `optimizer.state` with `SafeTensors.write`, and restore with `optimizer.restore(SafeTensors.open(path), steps: n)` (`GUIDE.md`, "Training").

8. **Build with `--release`.** Debug builds of tensor code are far slower: `crystal run --release main.cr` or `crystal build --release main.cr`. For CUDA, add `-Dcuda`.

## A complete program

`.claude/skills/train-a-model/example.cr` (`lib/tensomite/.claude/skills/train-a-model/example.cr`) trains a classifier on the CPU in seconds, prints a falling loss, saves and reloads the model, and tests it. Copy it into the project and run `crystal run --release example.cr`.

## Limits

- Computation is in `F32`, with `I32` for ids. `F16` and `BF16` weights widen to `F32` when loaded.
- Quantised weights (`Q8_0`, `Q4_0`, `Q4_1`) cannot be trained: gradients do not pass through them. Train in `F32`, then `model.quantize(DType::Q8_0)`.
- There is no tokeniser training. A text model uses characters, as `examples/shakespeare.cr` does, or an existing vocabulary loaded with `BPE.from_files` or `BPE.from_json`.
- Batching, shuffling, validation and learning-rate schedules are written by the user. `examples/shakespeare.cr` samples random windows and sets a warm-up and cosine decay through `optimizer.lr`; `apps/finetune/src/trainer.cr` adds validation, checkpoints and early stopping.
- Numbers go on the right of an operator (`x * 2`, not `2 * x`); both operands have the same dtype; a tensor holds at most about 2 GB.

## Where to read more

- `GUIDE.md` (`lib/tensomite/GUIDE.md`): the whole API with examples; start with "Neural-network building blocks", "Gradients", "Training", "Models from modules" and "Devices".
- `TUTORIAL.md` (`lib/tensomite/TUTORIAL.md`): a walk through `examples/multiply.cr`.
- `examples/shakespeare.cr` (`lib/tensomite/examples/shakespeare.cr`): a character-level GPT with batching, a schedule and validation.
- The `finetune-a-model` skill: fine-tuning a pretrained GPT-2.
