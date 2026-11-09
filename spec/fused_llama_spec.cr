require "./spec_helper"

# The composites the fused ops replace, built from primitives.
private def rmsnorm_composite(x : Tensor, weight : Tensor, eps : Float64) : Tensor
  x / ((x * x).mean(-1, keepdims: true) + eps).sqrt * weight
end

private def rope_composite(x : Tensor, cos : Tensor, sin : Tensor) : Tensor
  x * cos + x.rotate_half * sin
end

# Returns true when *device* computes the fused RMS norm as the host does; GPUs
# sum each row in chunks.
private def host_order?(device : Device) : Bool
  DEVICES.includes?(device)
end

# Returns the number of ops *device* has run on the host instead of its own
# kernels, or 0 for a device that runs everything on the host.
private def host_ops(device : Device) : Int32
  {% if flag?(:darwin) && flag?(:aarch64) %}
    return device.host_ops if device.is_a?(Metal)
  {% end %}
  {% if flag?(:cuda) %}
    return device.host_ops if device.is_a?(CUDA)
  {% end %}
  0
end

describe "Fused RMSNorm and RoPE" do
  x_values = random_values(6 * 2 * 96, 1)
  weight_values = random_values(96, 2)
  rope_values = random_values(2 * 4 * 3 * 96, 6)

  (DEVICES + GPUS).each do |device|
    it "computes RMSNorm as its composite on #{device.class}" do
      x = Tensor.from(x_values, [12, 96], device)
      weight = Tensor.from(weight_values, [96], device)
      fused = x.rmsnorm(weight, 1e-5)
      fused.node.op.should eq Op::RMSNorm
      expected = rmsnorm_composite(Tensor.from(x_values, [12, 96]), Tensor.from(weight_values, [96]), 1e-5).to_a(Float32)
      if host_order?(device)
        fused.to_a(Float32).should eq expected
      else
        fused.to_a(Float32).zip(expected) { |a, e| a.should be_close(e, 1e-5 * Math.max(1f32, e.abs)) }
      end
    end

    it "computes RoPE as its composite on #{device.class}, bit for bit, through views" do
      cos_values, sin_values = random_values(4 * 96, 3), random_values(4 * 96, 4)
      x = Tensor.from(rope_values, [2, 4, 3, 96], device).permute([0, 2, 1, 3])
      cos, sin = Tensor.from(cos_values, [4, 96], device), Tensor.from(sin_values, [4, 96], device)
      fused = x.rope(cos, sin)
      fused.node.op.should eq Op::Rope
      plain = Tensor.from(rope_values, [2, 4, 3, 96]).permute([0, 2, 1, 3]).contiguous
      expected = rope_composite(plain, Tensor.from(cos_values, [4, 96]), Tensor.from(sin_values, [4, 96])).to_a(Float32)
      fused.to_a(Float32).should eq expected
    end
  end

  GPUS.each do |gpu|
    it "let a quantised Llama decode on #{gpu.class} with no host op" do
      model = Llama.load("#{__DIR__}/fixtures/tiny_llama/model-q4_0.gguf", gpu)
      cache = model.new_cache
      model.forward([1, 2, 3, 4], cache)
      before = host_ops(gpu)
      model.forward([5], cache)
      host_ops(gpu).should eq before
    end
  end

  it "gives RMSNorm's gradients" do
    check_gradients([[3, 8], [8]]) { |(x, weight)| x.rmsnorm(weight) }
  end

  it "gives RoPE's gradient" do
    tables = random_values(2 * 4 * 8, 9)
    check_gradients([[2, 4, 8]]) do |(x)|
      x.rope(Tensor.from(tables[0, 32], [4, 8], x.device), Tensor.from(tables[32, 32], [4, 8], x.device))
    end
  end

  it "falls back to the composite for odd widths and integer data" do
    odd = Tensor.from(random_values(6, 4), [2, 3])
    odd.rmsnorm(Tensor.from(random_values(3, 5), [3])).node.op.should eq Op::RMSNorm
    expect_raises(ShapeError) { odd.rope(odd, odd) }
  end
end
