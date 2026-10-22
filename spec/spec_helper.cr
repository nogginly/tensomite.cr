require "spec"
require "../src/tensomite"

include Tensomite

def f32(*values : Number) : Array(Float32)
  values.map(&.to_f32).to_a
end
