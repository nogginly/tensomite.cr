require "./spec_helper"

# The composite `Tensor.attention` replaces: a matmul, a fused softmax and a matmul.
private def composite(q : Tensor, k : Tensor, v : Tensor, scale : Float64, mask : Tensor?) : Tensor
  q.matmul(k.transpose).softmax(scale: scale, mask: mask).matmul(v)
end

# Returns `[heads, rows, width]` queries and `[heads, positions, width]` keys and
# values, the keys and values as views of a larger cache, as decoding reads them.
private def inputs(device : Device, heads : Int32, rows : Int32, positions : Int32, width : Int32) : {Tensor, Tensor, Tensor}
  capacity = positions + 5
  q = Tensor.from(random_values(heads * rows * width, 1), [heads, rows, width], device)
  keys = Tensor.from(random_values(heads * capacity * width, 2), [heads, capacity, width], device)
  values = Tensor.from(random_values(heads * capacity * width, 3), [heads, capacity, width], device)
  {q, keys.slice(1, 0, positions), values.slice(1, 0, positions)}
end

# Returns a causal mask for the last *rows* of *positions*: row r sees positions
# up to `positions - rows + r`.
private def causal(rows : Int32, positions : Int32, device : Device) : Tensor
  values = Array.new(rows * positions) do |i|
    i % positions <= positions - rows + i // positions ? 0f32 : -Float32::INFINITY
  end
  Tensor.from(values, [rows, positions], device)
end

private def close(got : Array(Float32), expected : Array(Float32)) : Nil
  got.size.should eq expected.size
  got.zip(expected) { |a, e| a.should be_close(e, 1e-5 * Math.max(1f32, e.abs)) }
end

describe "Tensor.attention" do
  it "is the composite on host devices, bit for bit" do
    DEVICES.each do |device|
      q, k, v = inputs(device, 2, 3, 37, 16)
      mask = causal(3, 37, device)
      fused = Tensor.attention(q, k, v, scale: 0.25, mask: mask)
      fused.node.op.should_not eq Op::Attention
      fused.to_a(Float32).should eq composite(q, k, v, 0.25, mask).to_a(Float32)
    end
  end

  it "computes an Attention node a GPU leaves to the host" do
    q, k, v = inputs(CPU.default, 2, 3, 37, 16)
    mask = causal(3, 37, CPU.default)
    node = Node.new(Op::Attention, [q, k, v, mask.expand([2, 3, 37])], [2, 3, 16], DType::F32, CPU.default, scale: 0.25_f32)
    close(Tensor.lazy(node).to_a(Float32), composite(q, k, v, 0.25, mask).to_a(Float32))
  end

  GPUS.each do |gpu|
    [{1, 37, nil}, {3, 37, :causal}, {8, 300, :causal}, {1, 1, nil}].each do |(rows, positions, masked)|
      it "fuses on #{gpu.class}, #{rows} rows of #{positions} positions#{masked ? ", masked" : ""}, with the composite's bits where shape-invariant" do
        q, k, v = inputs(gpu, 2, rows, positions, 64)
        mask = masked ? causal(rows, positions, gpu) : nil
        fused = Tensor.attention(q, k, v, scale: 0.125, mask: mask)
        fused.node.op.should eq Op::Attention
        got = fused.to_a(Float32)
        same_device = composite(q, k, v, 0.125, mask).to_a(Float32)
        gpu.shape_invariant? ? got.should(eq(same_device)) : close(got, same_device)
        hq, hk, hv = inputs(CPU.default, 2, rows, positions, 64)
        close(got, composite(hq, hk, hv, 0.125, masked ? causal(rows, positions, CPU.default) : nil).to_a(Float32))
      end
    end

    it "keeps the composite on #{gpu.class} for more rows than it fuses" do
      q, k, v = inputs(gpu, 1, Tensor::ATTENTION_ROWS + 1, 20, 16)
      Tensor.attention(q, k, v, scale: 0.25).node.op.should_not eq Op::Attention
    end

    it "gives the composite's gradients on #{gpu.class}" do
      q, k, v = inputs(gpu, 2, 3, 37, 16)
      weights = Tensor.from(random_values(2 * 3 * 16, 4), [2, 3, 16], gpu)
      mask = causal(3, 37, gpu)
      fused = Tensor.attention(q, k, v, scale: 0.25, mask: mask)
      fused.node.op.should eq Op::Attention
      got = Tensomite.grad((fused * weights).sum, [q, k, v])
      expected = Tensomite.grad((composite(q, k, v, 0.25, mask) * weights).sum, [q, k, v])
      got.zip(expected) { |a, e| close(a.to_a(Float32), e.to_a(Float32)) }
    end
  end
end
