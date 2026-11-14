# Trains a small classifier to tell whether a point in the square [-1, 1]² lies
# inside the circle of radius 0.7, then saves it, loads it back and tests it.
# It trains on the CPU in a few seconds and prints a falling loss.
#
# In a project that depends on the tensomite shard, the require below finds it
# in `lib/tensomite`. To run this file inside the tensomite repository, copy it
# to the repository root and change the require to "./src/tensomite".
#
#     crystal run --release example.cr
require "tensomite"

include Tensomite

# Two inputs, two hidden layers of 32 units with GELU between them, and two
# output scores, one per class.
class Classifier < Tensomite::Module
  def initialize(build : Builder)
    @hidden1 = build.linear(2, 32)
    @hidden2 = build.linear(32, 32)
    @output = build.linear(32, 2)
  end

  # Returns lazy scores [batch, 2] for points [batch, 2].
  def forward(x : Tensor) : Tensor
    h = @hidden1.call(x).gelu
    h = @hidden2.call(h).gelu
    @output.call(h)
  end

  # Returns whether the model puts the point (*x*, *y*) inside the circle.
  def inside?(x : Float64, y : Float64) : Bool
    scores = call(Tensor.from([x.to_f32, y.to_f32], [1, 2])).to_a(Float32)
    scores[1] > scores[0]
  end
end

# Returns a batch of *size* random points and their classes: 1 inside the
# circle, 0 outside. Points are F32 and classes I32, as cross_entropy takes.
def batch(rng : Random, size : Int32) : {Tensor, Tensor}
  points = Array.new(size) { {rng.rand * 2 - 1, rng.rand * 2 - 1} }
  inputs = points.flat_map { |(x, y)| [x.to_f32, y.to_f32] }
  classes = points.map { |(x, y)| x * x + y * y < 0.49 ? 1 : 0 }
  {Tensor.from(inputs, [size, 2]), Tensor.from(classes, [size])}
end

rng = Random.new(42)
model = Classifier.new(Builder.new(rng: Random.new(1)))
optimizer = AdamW.new(model.params, lr: 5e-3, weight_decay: 0.0)

1000.times do |step|
  inputs, classes = batch(rng, 256)
  loss = optimizer.step(model.call(inputs).cross_entropy(classes))
  puts "step #{step}: loss #{loss.round(4)}" if step % 100 == 0 || step == 999
end

model.save("classifier.safetensors")
reloaded = Classifier.new(Builder.new(rng: Random.new(2))).load("classifier.safetensors")

correct = 0
1000.times do
  x, y = rng.rand * 2 - 1, rng.rand * 2 - 1
  correct += 1 if reloaded.inside?(x, y) == (x * x + y * y < 0.49)
end
puts "reloaded model: #{correct / 10}% of 1000 new points classified correctly"
