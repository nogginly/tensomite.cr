require "./spec_helper"

# The composites each spec kernel fuses, built from primitive ops: the oracles.
private def axpb(x : Tensor, a : Tensor, b : Tensor) : Tensor
  x * a + b
end

private def mixed(x : Tensor, y : Tensor) : Tensor
  (-x / (y * y + 1)).exp + (x * x + 1).log - (y * y + 1).sqrt
end

private def gelu_composite(x : Tensor) : Tensor
  inner = (x + x * x * x * 0.044715) * Math.sqrt(2 / Math::PI)
  x * 0.5 * (inner.tanh + 1)
end

private def random(device : Device, shape : Array(Int32), seed : Int32) : Tensor
  Tensor.from(random_values(shape.product(1), seed), shape, device)
end

describe Kernel do
  DEVICES.each do |device|
    describe "on #{device.class}" do
      it "matches its composite bit for bit, broadcasting included" do
        x, a, b = random(device, [3, 4], 1), random(device, [4], 2), random(device, [3, 1], 3)
        Kernels::SPEC_AXPB.call(x, a, b).to_a(Float32).should eq axpb(x, a, b).to_a(Float32)
      end

      it "matches its composite through transposed views" do
        x, y = random(device, [5, 3], 1).transpose, random(device, [3, 5], 2)
        Kernels::SPEC_MIXED.call(x, y).to_a(Float32).should eq mixed(x, y).to_a(Float32)
      end

      it "computes gelu exactly as the composite does" do
        x = Tensor.from(Array(Float32).new(200) { |i| ((i - 100) / 25).to_f32 }, [200], device)
        x.gelu.to_a(Float32).should eq gelu_composite(x).to_a(Float32)
      end

      it "computes square, sigmoid and rsqrt exactly as their composites do" do
        x = Tensor.from(Array(Float32).new(200) { |i| ((i - 100) / 25).to_f32 }, [200], device)
        x.square.to_a(Float32).should eq (x * x).to_a(Float32)
        x.sigmoid.to_a(Float32).should eq (Tensor.full([200], 1, device: device) / ((-x).exp + 1)).to_a(Float32)
        positive = x * x + 0.5
        positive.rsqrt.to_a(Float32).should eq (Tensor.full([200], 1, device: device) / positive.sqrt).to_a(Float32)
      end

      it "computes relu as max(0, x), with no gradient at 0" do
        x = Tensor.from(f32(-2, -0.5, 0, 0.5, 3), [5], device)
        x.relu.to_a(Float32).should eq f32(0, 0, 0, 0.5, 3)
        Tensomite.grad(x.relu.sum, [x]).first.to_a(Float32).should eq f32(0, 0, 0, 1, 1)
      end

      it "stacks along a new axis and unbinds back, through views" do
        a, b = random(device, [3, 4], 1), random(device, [4, 3], 2).transpose
        stacked = Tensor.stack([a, b], 1)
        stacked.shape.should eq [3, 2, 4]
        parts = stacked.unbind(1)
        parts.map(&.to_a(Float32)).should eq [a.to_a(Float32), b.to_a(Float32)]
        Tensor.stack([a, b], -1).shape.should eq [3, 4, 2]
        b.unbind(1).map(&.to_a(Float32)).should eq (0...4).map { |i| b.slice(1, i, i + 1).to_a(Float32) }
      end

      it "folds constant subexpressions in Float64, as scalar operands are" do
        x = random(device, [6], 1)
        Kernels::SPEC_FOLDED.call(x).to_a(Float32).should eq (x * (Math::PI / 2) - 1 / 3).to_a(Float32)
      end
    end
  end

  it "builds one node for the whole expression" do
    y = Tensor.from(f32(1, 2, 3), [3]).gelu
    y.node.op.should eq Op::Kernel
    y.node.srcs.map(&.node.op).should eq [nil]
  end

  it "differentiates each input" do
    check_gradients([[2, 3], [3], [2, 1]]) { |(x, a, b)| Kernels::SPEC_AXPB.call(x, a, b) }
    check_gradients([[2, 3], [2, 3]]) { |(x, y)| Kernels::SPEC_MIXED.call(x, y) }
    check_gradients([[2, 4]], &.first.gelu)
    check_gradients([[2, 4]], &.first.square)
    check_gradients([[2, 4]], &.first.sigmoid)
    check_gradients([[2, 4]]) { |(x)| (x * x + 1).rsqrt }
    check_gradients([[2, 3], [2, 3]]) { |(a, b)| Tensor.stack([a, b], 1) * Tensor.stack([b, a], 1) }
    check_gradients([[3, 4]]) { |(x)| x.unbind(0)[1] * x.unbind(1)[2].sum }
  end

  it "raises for stacking tensors of different shapes" do
    expect_raises(ShapeError, /shapes differ/) { Tensor.stack([Tensor.zeros([2]), Tensor.zeros([3])], 0) }
  end

  it "rejects the wrong number of inputs, and non-F32 inputs" do
    x = Tensor.from(f32(1, 2), [2])
    expect_raises(ArgumentError, /takes 3 inputs/) { Kernels::SPEC_AXPB.call(x, x) }
    expect_raises(DTypeError) { Tensor.from([1, 2], [2]).gelu }
  end

  it "writes MSL with one statement per operation" do
    msl = Kernels::GELU.msl
    msl.should contain "kernel void k_gelu("
    msl.should contain "precise::tanh("
    msl.lines.count(&.includes?(" = t")).should be > 3
  end
end
