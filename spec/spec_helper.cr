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
