# Teaches a small neural network to multiply two numbers between 0 and 10, then
# asks it some questions. TUTORIAL.md walks through it.
#
#     crystal run --release examples/multiply.cr
require "json"
require "../src/tensomite"

include Tensomite

# A multilayer perceptron: two inputs, two hidden layers of `HIDDEN` units with
# GELU between them, and one output.
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

  def initialize(@params : Array(Tensor))
  end

  # Returns predictions [batch, 1] for inputs [batch, 2], lazily.
  def forward(x : Tensor) : Tensor
    w1, b1, w2, b2, w3, b3 = params
    h = Tensomite.scope("hidden 1") { (x.matmul(w1) + b1).gelu }
    h = Tensomite.scope("hidden 2") { (h.matmul(w2) + b2).gelu }
    Tensomite.scope("output") { h.matmul(w3) + b3 }
  end

  # Returns the predicted product of *a* and *b*.
  def multiply(a : Float64, b : Float64) : Float64
    x = Tensor.from([(a / 10).to_f32, (b / 10).to_f32], [1, 2])
    forward(x).to_a(Float32).first.to_f64 * 100
  end

  # Returns the weights as JSON: each tensor's shape and values.
  def to_json : String
    params.map { |param| {shape: param.shape, values: param.to_a(Float32)} }.to_json
  end

  # Returns a model with the weights *json* holds, as written by `#to_json`.
  def self.from_json(json : String) : Multiplier
    tensors = Array(NamedTuple(shape: Array(Int32), values: Array(Float32))).from_json(json)
    new(tensors.map { |tensor| Tensor.from(tensor[:values], tensor[:shape]) })
  end
end

# Returns a batch of *size* random pairs from 0 to 10, scaled to [0, 1], and
# their products, scaled to [0, 1] too.
def batch(rng : Random, size : Int32) : {Tensor, Tensor}
  pairs = Array.new(size) { {rng.rand(10.0), rng.rand(10.0)} }
  inputs = pairs.flat_map { |(a, b)| [(a / 10).to_f32, (b / 10).to_f32] }
  targets = pairs.map { |(a, b)| (a * b / 100).to_f32 }
  {Tensor.from(inputs, [size, 2]), Tensor.from(targets, [size, 1])}
end

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

puts
[{3.0, 4.0}, {7.0, 8.0}, {9.5, 2.0}, {0.5, 0.5}, {12.0, 12.0}].each do |(a, b)|
  puts "#{a} × #{b} = #{(a * b).round(2)}, model says #{model.multiply(a, b).round(2)}"
end

File.write("multiplier.json", model.to_json)
reloaded = Multiplier.from_json(File.read("multiplier.json"))
puts
puts "reloaded from multiplier.json: 3 × 4 = #{reloaded.multiply(3.0, 4.0).round(2)}"

puts
puts model.forward(Tensor.zeros([1, 2]).named("a, b")).to_mermaid(depth: 1, params: model.params)
