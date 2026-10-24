# Tutorial: teaching a network to multiply

This tutorial trains a small neural network to multiply two numbers between 0 and 10, then uses it for inference: asking it questions it has not seen, saving it, and loading it back. It uses only the CPU and takes about 15 seconds to train. The whole program is [`examples/multiply.cr`](examples/multiply.cr); run it with:

```sh
crystal run --release examples/multiply.cr
```

The [user's guide](GUIDE.md) explains each part of the API used here.

## 1. The problem

We want a function `f(a, b) ≈ a × b` for `a` and `b` between 0 and 10, learned from examples rather than written down. A network does not know what multiplication is. It sees many pairs and their products and adjusts its weights until its answers are close. That makes multiplication a good first task: the right answer is never in doubt, so it is easy to see how well the network has learned, and where it fails.

**Scaling the data.** Networks learn best when inputs and outputs are small numbers, roughly between -1 and 1. So the inputs are `a / 10` and `b / 10`, both in `[0, 1]`, and the target is `a × b / 100`, also in `[0, 1]`. To read an answer, multiply the output by 100.

```crystal
# Returns a batch of *size* random pairs from 0 to 10, scaled to [0, 1], and
# their products, scaled to [0, 1] too.
def batch(rng : Random, size : Int32) : {Tensor, Tensor}
  pairs = Array.new(size) { {rng.rand(10.0), rng.rand(10.0)} }
  inputs = pairs.flat_map { |(a, b)| [(a / 10).to_f32, (b / 10).to_f32] }
  targets = pairs.map { |(a, b)| (a * b / 100).to_f32 }
  {Tensor.from(inputs, [size, 2]), Tensor.from(targets, [size, 1])}
end
```

Each batch is fresh random pairs, so the network never sees the same question twice. The inputs are a `[256, 2]` tensor, one row per pair, and the targets `[256, 1]`.

## 2. The model

The network is a **multilayer perceptron**: layers that each multiply by a weight matrix, add a bias, and pass the result through a non-linear function, here `gelu`.

```mermaid
flowchart LR
    In["a/10, b/10<br/>[batch, 2]"] --> L1["matmul w1 + b1, gelu<br/>[batch, 64]"]
    L1 --> L2["matmul w2 + b2, gelu<br/>[batch, 64]"]
    L2 --> Out["matmul w3 + b3<br/>[batch, 1]"]
    Out --> Answer["× 100 = a × b"]
```

Why the non-linear function? Without it, three matrix products in a row equal a single matrix product, which can only compute a linear function of `a` and `b`, of the form `p·a + q·b + r`. `a × b` is not linear, so the network needs `gelu` between the layers to represent it.

```crystal
class Multiplier
  HIDDEN = 64

  # The weights and biases, in the order `#forward` uses them.
  property params : Array(Tensor)

  def initialize(seed : Int32)
    rng = Random.new(seed)
    # Returns a [rows, cols] matrix of small random weights, scaled by 1/√rows
    # so that each layer's outputs start at about the size of its inputs.
    weight = ->(rows : Int32, cols : Int32) do
      scale = 1 / Math.sqrt(rows)
      Tensor.from(Array(Float32).new(rows * cols) { (rng.rand(-1.0..1.0) * scale).to_f32 }, [rows, cols])
    end
    bias = ->(size : Int32) { Tensor.zeros([size]) }
    @params = [weight.call(2, HIDDEN), bias.call(HIDDEN),
               weight.call(HIDDEN, HIDDEN), bias.call(HIDDEN),
               weight.call(HIDDEN, 1), bias.call(1)]
  end

  # Returns predictions [batch, 1] for inputs [batch, 2], lazily.
  def forward(x : Tensor) : Tensor
    w1, b1, w2, b2, w3, b3 = params
    h = Tensomite.scope("hidden 1") { (x.matmul(w1) + b1).gelu }
    h = Tensomite.scope("hidden 2") { (h.matmul(w2) + b2).gelu }
    Tensomite.scope("output") { h.matmul(w3) + b3 }
  end
end
```

Three details:

1. **Random starting weights.** If every weight began equal, every hidden unit would compute the same value and receive the same gradient, so they would stay identical. Random weights make them differ from the start. The `1/√rows` scale keeps the numbers from growing or shrinking layer by layer.
2. **Broadcasting does the bias.** `x.matmul(w1)` is `[batch, 64]` and `b1` is `[64]`, so `+ b1` adds the bias to every row.
3. **`Tensomite.scope`** names each layer. It changes nothing in the arithmetic; it is for the diagram in step 7.

## 3. The loss

The loss is one number that says how wrong the network is: here the **mean squared error**, the average of `(prediction - target)²` over the batch. Squaring makes every miss positive, and big misses count for more than small ones.

```crystal
error = model.forward(inputs) - targets
loss = (error * error).mean
```

## 4. The training loop

Training repeats four steps. Each asks the loss how it would change if each weight changed a little (the gradient), then moves every weight a little in the direction that lowers it.

```crystal
rng = Random.new(42)
model = Multiplier.new(seed: 1)
optimizer = AdamW.new(lr: 3e-3, weight_decay: 0.0)

2000.times do |step|
  inputs, targets = batch(rng, 256)
  error = model.forward(inputs) - targets
  loss = (error * error).mean
  grads = Tensomite.grad(loss, model.params)
  loss_value = loss.to_a(Float32).first
  model.params = optimizer.step(model.params, grads)
  puts "step #{step}: loss #{loss_value.round(6)}" if step % 250 == 0 || step == 1999
end
```

1. **Forward:** predictions for a fresh batch. Nothing is computed yet: tensomite is lazy, and this only builds a graph.
2. **Gradients:** `Tensomite.grad` extends the graph backwards, from the loss to every parameter. It comes before `loss.to_a`, because reading the loss runs the graph and frees the intermediate results that the gradients need.
3. **Read the loss,** for the log. This runs the forward graph.
4. **Step:** `AdamW` computes new parameters from the gradients, which runs the rest of the graph. Tensors never change in place, so the model keeps the new ones.

`AdamW` adapts the step size of each weight to the history of its gradients, so it works well over a wider range of learning rates (`lr`) than plain gradient descent. Weight decay, which shrinks every weight slightly at each step and helps large models generalise, is switched off: a model this small does not need it.

## 5. Watching it learn

```
step 0: loss 0.11273
step 250: loss 3.4e-5
step 500: loss 5.0e-6
step 750: loss 4.0e-6
step 1000: loss 4.0e-6
step 1250: loss 3.0e-6
step 1500: loss 3.0e-6
step 1750: loss 2.0e-6
step 1999: loss 2.0e-6
```

The loss falls by a factor of 3,000 in the first 250 steps, then falls slowly. A loss of 2e-6 on the scaled targets is a root-mean-square error of about `√(2e-6) × 100 ≈ 0.14` in the products: on average, the network is off by about a seventh.

## 6. Asking questions

Inference is the forward pass alone, on one pair at a time, scaled in and out:

```crystal
# Returns the predicted product of *a* and *b*.
def multiply(a : Float64, b : Float64) : Float64
  x = Tensor.from([(a / 10).to_f32, (b / 10).to_f32], [1, 2])
  forward(x).to_a(Float32).first.to_f64 * 100
end
```

```
3.0 × 4.0 = 12.0, model says 11.96
7.0 × 8.0 = 56.0, model says 56.26
9.5 × 2.0 = 19.0, model says 18.93
0.5 × 0.5 = 0.25, model says -0.12
12.0 × 12.0 = 144.0, model says 137.92
```

The first three are good: pairs it never saw, answered to within a few tenths. The last two show what the network has really learned:

- **`0.5 × 0.5` comes out negative.** The loss counts *absolute* errors, so an error of 0.37 costs the same whether the true product is 0.25 or 56, though it is large relative to 0.25 and small relative to 56. The network has no reason to be precise about small products. Training on the error relative to the product would change that.
- **`12 × 12` is outside the training range,** and the answer is off by 6. The network learned to approximate `a × b` for inputs between 0 and 10, not the rule of multiplication. Outside that range nothing was trained, and its answers become less accurate the further out the inputs are. This is the main lesson of the tutorial: a network is reliable within the range of its training data, and not outside it.

## 7. Saving, loading and drawing the model

Tensomite cannot write weight files yet, but a parameter is just a shape and a list of numbers, so JSON does the job:

```crystal
# Returns the weights as JSON: each tensor's shape and values.
def to_json : String
  params.map { |param| {shape: param.shape, values: param.to_a(Float32)} }.to_json
end

# Returns a model with the weights *json* holds, as written by `#to_json`.
def self.from_json(json : String) : Multiplier
  tensors = Array(NamedTuple(shape: Array(Int32), values: Array(Float32))).from_json(json)
  new(tensors.map { |tensor| Tensor.from(tensor[:values], tensor[:shape]) })
end
```

```
reloaded from multiplier.json: 3 × 4 = 11.96
```

The reloaded model gives exactly the same answer: the weights are the same bits, and tensomite's results repeat exactly.

Finally, the scopes from step 2 let tensomite draw the model, one box per layer with its parameter count:

```crystal
puts model.forward(Tensor.zeros([1, 2]).named("a, b")).to_mermaid(depth: 1, params: model.params)
```

```mermaid
flowchart TB
    b0["a, b · leaf [1,2]"]
    b1["hidden 1<br/>3 ops · 192 params"]
    b2["hidden 2<br/>3 ops · 4.16k params"]
    b3["output<br/>2 ops · 65 params"]
    b0 -- "[1,2]" --> b1
    b1 -- "[1,64]" --> b2
    b2 -- "[1,64]" --> b3
    classDef leaf stroke:#2b8a3e,stroke-width:2px
    class b0 leaf
    classDef root stroke:#c92a2a,stroke-width:2px
    class b3 root
```

The counts check out by hand: the first layer has a `2 × 64` weight and 64 biases (192), the second `64 × 64 + 64` (4,160), and the output `64 × 1 + 1` (65). That is 4,417 numbers in all, to multiply two numbers. Multiplication is only a test case: the same training loop, with a larger model and more data, is how models learn tasks that have no simple formula.

## Things to try

1. **Relative error.** Divide each error by its target plus a small constant, such as `targets + 0.01`, before squaring. Do small products improve, and large ones get worse?
2. **A wider range.** Train on pairs from 0 to 20, and ask `12 × 12` again.
3. **A smaller model.** Set `HIDDEN` to 8. How low does the loss go now?
4. **Speed.** Add `require "../src/tensomite/blas"` and create the tensors on `BLAS.default` (each `Tensor.from` and `Tensor.zeros` takes a `device:` argument). The model is too small for it to help much; try it with `HIDDEN = 512`.
5. **Something new.** Change `batch` to teach another function: `a + b`, `max(a, b)`, or `sin(a)`. Which are easy to learn, and which are hard?

For a larger model trained the same way, see `examples/shakespeare.cr`, which learns to write in the style of Shakespeare one character at a time.
