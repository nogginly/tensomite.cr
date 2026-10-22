require "spec"
require "../src/tensomite"

include Tensomite

# Devices every op spec runs on; results must agree exactly.
DEVICES = [CPU.default, SimDiscrete.default] of Device

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
