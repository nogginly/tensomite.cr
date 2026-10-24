require "spec"
require "../src/tensomite"
require "../src/tensomite/blas"
{% if flag?(:darwin) && flag?(:aarch64) %}
  require "../src/tensomite/metal"
{% end %}

include Tensomite

# Devices every op spec runs on; results must agree exactly.
DEVICES = [CPU.default, SimDiscrete.default, BLAS.default] of Device

# GPU devices present on this machine, compared with the CPU in `gpu_spec.cr`.
# Empty where there is none, such as on Linux or a hosted macOS runner.
GPUS = {% if flag?(:darwin) && flag?(:aarch64) %}
         Metal.available? ? [Metal.default] of Device : [] of Device
       {% else %}
         [] of Device
       {% end %}

def f32(*values : Number) : Array(Float32)
  values.map(&.to_f32).to_a
end

def tensor(device : Device, shape : Array(Int32), data : Array(T)) : Tensor forall T
  Tensor.from(data, shape, device)
end

# Checks *actual* against *expected* element by element, within *tolerance*.
def expect_close(actual : Array(Float32), expected : Array(Float32), tolerance = 1e-6) : Nil
  actual.size.should eq expected.size
  actual.zip(expected) { |a, e| a.should be_close(e, tolerance) }
end

# Returns *n* values in [-1, 1) from a fixed seed.
def random_values(n : Int32, seed : Int32) : Array(Float32)
  rng = Random.new(seed)
  Array.new(n) { rng.rand(-1.0..1.0).to_f32 }
end

# Returns a copy of *data* with element *j* of input *i* moved by *delta*.
def nudge(data : Array(Array(Float32)), i : Int32, j : Int32, delta : Float64) : Array(Array(Float32))
  data.map(&.dup).tap { |copy| copy[i][j] += delta }
end

# Checks the gradients `Tensomite.grad` gives for `f(inputs)` against central
# finite differences, on every device.
#
# The loss is the sum of `f`'s output times fixed random weights, so that every
# output element contributes differently. *shapes* gives the input shapes; inputs
# are random unless *data* supplies them.
def check_gradients(shapes : Array(Array(Int32)), data : Array(Array(Float32))? = nil,
                    eps = 1e-2, tolerance = 2e-3, &f : Array(Tensor) -> Tensor) : Nil
  data ||= shapes.map_with_index { |shape, i| random_values(shape.product(1), 7 + i) }
  DEVICES.each do |device|
    loss_of = ->(xs : Array(Array(Float32))) do
      inputs = xs.zip(shapes).map { |(x, shape)| Tensor.from(x, shape, device) }
      output = f.call(inputs)
      weights = Tensor.from(random_values(output.size, 99), output.shape, device)
      {inputs, (output * weights).sum}
    end
    inputs, loss = loss_of.call(data)
    analytic = Tensomite.grad(loss, inputs).map(&.to_a(Float32))
    data.each_with_index do |x, i|
      x.size.times do |j|
        up = nudge(data, i, j, eps)
        down = nudge(data, i, j, -eps)
        numeric = (loss_of.call(up)[1].to_a(Float32)[0] - loss_of.call(down)[1].to_a(Float32)[0]) / (2 * eps)
        analytic[i][j].should be_close(numeric, tolerance * (1 + numeric.abs))
      end
    end
  end
end

# Kernels used only by the specs, defined here so that every spec file sees them.
Tensomite.kernel(:spec_axpb) { |x, a, b| x * a + b }
Tensomite.kernel(:spec_mixed) { |x, y| (-x / (y * y + 1)).exp + (x * x + 1).log - (y * y + 1).sqrt }
Tensomite.kernel(:spec_folded) { |x| x * (Math::PI / 2) - 1 / 3 }
