require "./spec_helper"

# Returns a tensor of *shape* with seeded random values in [lo, hi) on *device*.
private def random(device : Device, shape : Array(Int32), seed : Int32, lo = -1f32, hi = 1f32) : Tensor
  rng = Random.new(seed)
  Tensor.from(Array(Float32).new(shape.product(1)) { rng.rand(lo...hi) }, shape, device)
end

# Ops whose GPU results must equal the CPU's bit for bit: IEEE arithmetic and
# comparisons are exact whatever the device. Inputs cover dense, transposed,
# broadcast and sliced views.
EXACT = {
  "add"            => ->(d : Device) { random(d, [5, 7], 1) + random(d, [7], 2) },
  "sub"            => ->(d : Device) { random(d, [7, 5], 1).transpose - random(d, [5, 1], 2) },
  "mul"            => ->(d : Device) { random(d, [5, 9], 1).slice(1, 2, 9) * random(d, [5, 7], 2) },
  "maximum"        => ->(d : Device) { random(d, [3, 4, 5], 1).maximum(random(d, [4, 5], 2)) },
  "greater_equal"  => ->(d : Device) { random(d, [6, 6], 1).greater_equal(random(d, [6, 6], 2).transpose) },
  "copy"           => ->(d : Device) { random(d, [4, 5, 6], 1).permute([2, 0, 1]).contiguous },
  "scalar"         => ->(d : Device) { random(d, [5, 5], 1) * 0.5 - 2 },
  "sum"            => ->(d : Device) { random(d, [6, 7], 1).transpose.sum(1) },
  "sum over all"   => ->(d : Device) { random(d, [3, 4, 5], 1).slice(2, 1, 4).sum },
  "max"            => ->(d : Device) { random(d, [6, 7], 1).max(0, keepdims: true) },
  "max over all"   => ->(d : Device) { random(d, [4, 6], 1).transpose.max },
  "cat"            => ->(d : Device) { Tensor.cat([random(d, [2, 3], 1), random(d, [4, 3], 2)], 0) },
  "cat of views"   => ->(d : Device) { Tensor.cat([random(d, [3, 2], 1).transpose, random(d, [2], 2).reshape([2, 1]).expand([2, 3])], 1) },
  "gather"         => ->(d : Device) { random(d, [5, 4], 1).gather(Tensor.from([4, 0, 2, 2], [4], d)) },
  "gather 2-D ids" => ->(d : Device) { random(d, [6, 3], 1).transpose.contiguous.transpose.gather(Tensor.from([5, 1, 0, 5], [2, 2], d)) },
  "scatter_add"    => ->(d : Device) {
    table = random(d, [5, 3], 1)
    Tensomite.grad((table.gather(Tensor.from([1, 3, 1, 4, 1], [5], d)) * random(d, [5, 3], 2)).sum, [table]).first
  },
  "unview of expand" => ->(d : Device) {
    bias = random(d, [4], 1)
    Tensomite.grad(((random(d, [3, 4], 2) + bias) * random(d, [3, 4], 3)).sum, [bias]).first
  },
  "unview of slice and transpose" => ->(d : Device) {
    x = random(d, [4, 6], 1)
    Tensomite.grad((x.transpose.slice(0, 1, 5) * random(d, [4, 4], 2)).sum, [x]).first
  },
  "unview of a broadcast batch" => ->(d : Device) {
    x = random(d, [3, 1, 2], 1)
    Tensomite.grad((x.expand([3, 4, 2]).permute([1, 2, 0]) * random(d, [4, 2, 3], 2)).sum, [x]).first
  },
  "matmul" => ->(d : Device) { random(d, [2, 5, 6], 1).matmul(random(d, [6, 3], 2)) },
}

# Ops whose GPU results may differ in the last bits: division and transcendental
# functions, whose rounding is the GPU maths library's (Decision K, level 3).
CLOSE = {
  "div"       => ->(d : Device) { random(d, [5, 7], 1) / random(d, [5, 7], 2, 0.5f32, 2f32) },
  "exp"       => ->(d : Device) { random(d, [5, 7], 1, -4f32, 4f32).exp },
  "log"       => ->(d : Device) { random(d, [7, 5], 1, 0.01f32, 10f32).transpose.log },
  "tanh"      => ->(d : Device) { random(d, [5, 7], 1, -3f32, 3f32).tanh },
  "sqrt"      => ->(d : Device) { random(d, [5, 7], 1, 0f32, 10f32).sqrt },
  "softmax"   => ->(d : Device) { random(d, [4, 9], 1, -5f32, 5f32).softmax },
  "layernorm" => ->(d : Device) { random(d, [4, 8], 1).layernorm(random(d, [8], 2), random(d, [8], 3)) },
  "gelu"      => ->(d : Device) { random(d, [4, 8], 1, -3f32, 3f32).gelu },
  "gradient"  => ->(d : Device) {
    w = random(d, [4, 3], 1)
    Tensomite.grad((random(d, [2, 4], 2).matmul(w).tanh * 3).sum, [w]).first
  },
}

describe "GPU devices against the CPU" do
  pending "no GPU on this machine" if GPUS.empty?

  GPUS.each do |gpu|
    describe gpu.class do
      EXACT.each do |name, build|
        it "matches #{name} exactly" do
          actual = build.call(gpu)
          expected = build.call(CPU.default)
          actual.shape.should eq expected.shape
          actual.to_a(Float32).should eq expected.to_a(Float32)
        end
      end

      CLOSE.each do |name, build|
        it "matches #{name} within 1e-6 relative error" do
          actual = build.call(gpu).to_a(Float32)
          expected = build.call(CPU.default).to_a(Float32)
          actual.size.should eq expected.size
          actual.zip(expected) { |a, e| a.should be_close(e, 1e-6 * Math.max(1f32, e.abs)) }
        end
      end

      it "raises for an id out of range when realised" do
        table = Tensor.from(f32(1, 2, 3, 4), [2, 2], gpu)
        result = table.gather(Tensor.from([0, 2], [2], gpu))
        expect_raises(ExecutionError, /index 2 out of range for 2 rows/) { result.realize }
        (table * 2).to_a(Float32).should eq f32(2, 4, 6, 8)
      end

      it "reads data back through map, download and to" do
        t = Tensor.from(f32(1, 2, 3), [3], gpu) * 2
        t.to_a(Float32).should eq f32(2, 4, 6)
        t.to(CPU.default).to_a(Float32).should eq f32(2, 4, 6)
        Tensor.from(f32(5, 6), [2]).to(gpu).to_a(Float32).should eq f32(5, 6)
      end
    end
  end
end
