# Teaches a small neural network to multiply two numbers between 0 and 10, then
# asks it some questions. TUTORIAL.md walks through it.
#
#     crystal run --release examples/multiply.cr
require "json"
require "../src/tensomite"

include Tensomite

# A multilayer perceptron: two inputs, two hidden layers of 64 units with GELU
# between them, and one output.
class Multiplier < Tensomite::Module
  def initialize(rng : Random)
    @hidden1 = Linear.new(2, 64, rng: rng)
    @hidden2 = Linear.new(64, 64, rng: rng)
    @output = Linear.new(64, 1, rng: rng)
  end

  # Returns predictions [batch, 1] for inputs [batch, 2], lazily.
  def forward(x : Tensor) : Tensor
    h = @hidden1.call(x).gelu
    h = @hidden2.call(h).gelu
    @output.call(h)
  end

  # Returns the predicted product of *a* and *b*.
  def multiply(a : Float64, b : Float64) : Float64
    x = Tensor.from([(a / 10).to_f32, (b / 10).to_f32], [1, 2])
    call(x).to_a(Float32).first.to_f64 * 100
  end

  # Returns the parameters as JSON, by name: each one's shape and values.
  def to_json : String
    named_params.transform_values { |param| {shape: param.shape, values: param.value.to_a(Float32)} }.to_json
  end

  # Replaces each parameter with the one of the same name in *json*, as
  # written by `#to_json`.
  def load_json(json : String) : self
    saved = Hash(String, NamedTuple(shape: Array(Int32), values: Array(Float32))).from_json(json)
    named_params.each { |name, param| param.value = Tensor.from(saved[name][:values], saved[name][:shape]) }
    self
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
model = Multiplier.new(rng: Random.new(1))
optimizer = AdamW.new(model.params, lr: 3e-3, weight_decay: 0.0)

2000.times do |step|
  inputs, targets = batch(rng, 256)
  error = model.call(inputs) - targets
  loss = optimizer.step((error * error).mean)
  puts "step #{step}: loss #{loss.round(6)}" if step % 250 == 0 || step == 1999
end

puts
[{3.0, 4.0}, {7.0, 8.0}, {9.5, 2.0}, {0.5, 0.5}, {12.0, 12.0}].each do |(a, b)|
  puts "#{a} × #{b} = #{(a * b).round(2)}, model says #{model.multiply(a, b).round(2)}"
end

File.write("multiplier.json", model.to_json)
reloaded = Multiplier.new(rng: Random.new(2)).load_json(File.read("multiplier.json"))
puts
puts "reloaded from multiplier.json: 3 × 4 = #{reloaded.multiply(3.0, 4.0).round(2)}"

puts
puts model.call(Tensor.zeros([1, 2]).named("a, b")).to_mermaid(model: model, depth: 1)
