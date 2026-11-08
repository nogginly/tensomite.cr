require "./spec_helper"

# Attention of one batch row computed head by head, in Float64, each query head
# using key and value head `head // group`, as Hugging Face's `repeat_kv` does.
private def reference_attention(q : Array(Float32), k : Array(Float32), v : Array(Float32),
                                heads : Int32, kv_heads : Int32, length : Int32, width : Int32) : Array(Float64)
  group = heads // kv_heads
  out = Array(Float64).new(length * heads * width, 0.0)
  heads.times do |head|
    kv = head // group
    length.times do |i|
      scores = (0..i).map do |j|
        (0...width).sum { |dim| q[(i * heads + head) * width + dim].to_f64 * k[(j * kv_heads + kv) * width + dim] } / Math.sqrt(width)
      end
      top = scores.max
      exps = scores.map { |score| Math.exp(score - top) }
      total = exps.sum
      width.times do |dim|
        out[(i * heads + head) * width + dim] = (0..i).sum { |j| exps[j] / total * v[(j * kv_heads + kv) * width + dim] }
      end
    end
  end
  out
end

describe "Llama's ops" do
  it "computes SiLU and SwiGLU's gate" do
    xs = f32(-3, -0.5, 0, 0.5, 3)
    ups = f32(2, -1, 4, 0.25, -0.5)
    DEVICES.each do |device|
      x = tensor(device, [5], xs)
      expect_close x.silu.to_a(Float32), xs.map { |e| e / (1 + Math.exp(-e)) }.map(&.to_f32)
      expect_close x.swiglu(tensor(device, [5], ups)).to_a(Float32),
        xs.zip(ups).map { |value, gate| value / (1 + Math.exp(-value)) * gate }.map(&.to_f32)
    end
  end

  it "gives SiLU and SwiGLU gradients" do
    check_gradients([[6]]) { |(x)| x.silu }
    check_gradients([[6], [6]]) { |(g, u)| g.swiglu(u) }
  end

  it "computes RMSNorm" do
    xs = f32(1, -2, 3, 0.5, 4, -1)
    ws = f32(1, 0.5, 2)
    DEVICES.each do |device|
      got = tensor(device, [2, 3], xs).rmsnorm(tensor(device, [3], ws), 1e-5).to_a(Float32)
      expected = xs.each_slice(3).to_a.flat_map do |row|
        scale = 1 / Math.sqrt(row.sum { |e| e.to_f64 ** 2 } / 3 + 1e-5)
        row.zip(ws).map { |value, weight| (value * scale * weight).to_f32 }
      end
      expect_close got, expected
    end
    check_gradients([[2, 4], [4]]) { |(x, w)| x.rmsnorm(w) }
  end

  it "rotates pairs of features by position" do
    rope = RotaryEmbedding.new(4, context: 8, theta: 100.0)
    xs = random_values(3 * 4, 3)
    DEVICES.each do |device|
      rope = RotaryEmbedding.new(4, context: 8, theta: 100.0, device: device)
      got = rope.call(tensor(device, [1, 3, 4], xs), start: 2).to_a(Float32)
      expected = (0...3).flat_map do |row|
        position = row + 2
        (0...4).map do |i|
          angle = position / 100.0 ** ((2 * (i % 2)).to_f64 / 4)
          pair = i < 2 ? -xs[row * 4 + i + 2] : xs[row * 4 + i - 2]
          (xs[row * 4 + i] * Math.cos(angle) + pair * Math.sin(angle)).to_f32
        end
      end
      expect_close got, expected, 1e-5
    end
    expect_raises(ShapeError) { rope.call(Tensor.zeros([1, 2, 4]), start: 7) }
  end

  (DEVICES + GPUS).each do |device|
    it "matches the CPU for SiLU, SwiGLU, RMSNorm and RoPE on #{device.class}" do
      results = [CPU.default, device].map do |dev|
        x = Tensor.from(random_values(4 * 8, 1), [1, 4, 8], dev)
        up = Tensor.from(random_values(4 * 8, 2), [1, 4, 8], dev)
        weight = Tensor.from(random_values(8, 3), [8], dev)
        rope = RotaryEmbedding.new(8, context: 8, device: dev)
        [x.silu, x.swiglu(up), x.rmsnorm(weight), rope.call(x, 3)].map(&.to_a(Float32))
      end
      results[1].zip(results[0]) do |actual, expected|
        actual.zip(expected) { |a, e| a.should be_close(e, 1e-5 * Math.max(1f32, e.abs)) }
      end
    end
  end

  describe GroupedQueryAttention do
    width, heads, kv_heads, length = 12, 6, 2, 5
    head_width = width // heads

    it "attends each query head to its group's keys and values" do
      rng = Random.new(5)
      attention = GroupedQueryAttention.new(width, heads, kv_heads, rng: rng, init: Init.normal(0.5))
      rope = RotaryEmbedding.new(head_width, context: 16)
      x = Tensor.from(random_values(length * width, 9), [length, width])
      mask = Tensor.causal_mask(length, length)
      got = attention.call(x, mask, rope).to_a(Float32)

      split = ->(t : Tensor, count : Int32) { t.reshape([1, length, count, head_width]).permute([0, 2, 1, 3]) }
      join = ->(t : Tensor) { t.permute([0, 2, 1, 3]).contiguous.to_a(Float32) }
      q = join.call(rope.call(split.call(attention.q_proj.call(x), heads)))
      k = join.call(rope.call(split.call(attention.k_proj.call(x), kv_heads)))
      v = join.call(split.call(attention.v_proj.call(x), kv_heads))
      attended = reference_attention(q, k, v, heads, kv_heads, length, head_width).map(&.to_f32)
      expected = attention.o_proj.call(Tensor.from(attended, [length, width])).to_a(Float32)
      expect_close got, expected, 1e-5
    end

    (DEVICES + GPUS).select(&.shape_invariant?).each do |device|
      it "gives the same bits in prefill and decode on #{device.class}" do
        attention = GroupedQueryAttention.new(width, heads, kv_heads, rng: Random.new(5), device: device)
        rope = RotaryEmbedding.new(head_width, context: 16, device: device)
        x = Tensor.from(random_values(length * width, 9), [length, width], device)
        whole = attention.call(x, Tensor.causal_mask(length, length, device), rope).to_a(Float32)

        cache = CausalSelfAttention::Cache.new(16)
        prefix = attention.call(x.slice(0, 0, 3), Tensor.causal_mask(3, 3, device), rope, cache).realize
        rows = [prefix.to_a(Float32)]
        (3...length).each do |i|
          rows << attention.call(x.slice(0, i, i + 1), Tensor.causal_mask(1, i + 1, device), rope, cache).to_a(Float32)
        end
        rows.flatten.should eq whole
        on_cpu = GroupedQueryAttention.new(width, heads, kv_heads, rng: Random.new(5)).call(
          Tensor.from(random_values(length * width, 9), [length, width]), Tensor.causal_mask(length, length),
          RotaryEmbedding.new(head_width, context: 16)).to_a(Float32)
        if DEVICES.includes?(device)
          whole.should eq on_cpu
        else
          whole.zip(on_cpu) { |a, e| a.should be_close(e, 1e-5 * Math.max(1f32, e.abs)) }
        end
      end
    end

    it "gives gradients" do
      check_gradients([[3, width]]) do |(x)|
        attention = GroupedQueryAttention.new(width, heads, kv_heads, rng: Random.new(2), device: x.device)
        rope = RotaryEmbedding.new(head_width, context: 8, device: x.device)
        attention.call(x, Tensor.causal_mask(3, 3, x.device), rope)
      end
    end

    it "rejects heads that do not group" do
      expect_raises(ShapeError) { GroupedQueryAttention.new(12, 6, 4) }
    end
  end

  it "runs SwiGLU's MLP" do
    mlp = SwiGLU.new(4, 6, rng: Random.new(1))
    x = Tensor.from(random_values(8, 4), [2, 4])
    expected = mlp.down_proj.call(mlp.gate_proj.call(x).silu * mlp.up_proj.call(x)).to_a(Float32)
    expect_close mlp.call(x).to_a(Float32), expected
    mlp.named_params.keys.should eq ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
  end
end
