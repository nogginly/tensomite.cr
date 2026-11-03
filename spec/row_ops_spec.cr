require "./spec_helper"

# The composites the fused row ops replace, built from primitive ops: the oracles.
private def softmax_composite(x : Tensor) : Tensor
  e = (x - x.max(-1, keepdims: true)).exp
  e / e.sum(-1, keepdims: true)
end

private def log_softmax_composite(x : Tensor) : Tensor
  shifted = x - x.max(-1, keepdims: true)
  shifted - shifted.exp.sum(-1, keepdims: true).log
end

private def layernorm_composite(x : Tensor, weight : Tensor, bias : Tensor) : Tensor
  centred = x - x.sum(-1, keepdims: true) / x.shape[-1]
  variance = (centred * centred).sum(-1, keepdims: true) / x.shape[-1]
  centred / (variance + 1e-5).sqrt * weight + bias
end

private def random(device : Device, shape : Array(Int32), seed : Int32, scale = 3f32) : Tensor
  Tensor.from(random_values(shape.product(1), seed).map { |v| v * scale }, shape, device)
end

describe "Fused row ops" do
  DEVICES.each do |device|
    describe "on #{device.class}" do
      it "computes softmax exactly as the composite does" do
        x = random(device, [5, 7], 1)
        x.softmax.to_a(Float32).should eq softmax_composite(x).to_a(Float32)
        x.transpose.softmax.to_a(Float32).should eq softmax_composite(x.transpose).to_a(Float32)
      end

      it "computes log_softmax exactly as the composite does" do
        x = random(device, [4, 9], 2).slice(1, 2, 8)
        x.log_softmax.to_a(Float32).should eq log_softmax_composite(x).to_a(Float32)
      end

      it "computes layernorm exactly as the composite does" do
        x, w, b = random(device, [3, 4, 8], 3), random(device, [8], 4), random(device, [8], 5)
        x.layernorm(w, b).to_a(Float32).should eq layernorm_composite(x, w, b).to_a(Float32)
        xt = x.permute([1, 0, 2])
        xt.layernorm(w, b).to_a(Float32).should eq layernorm_composite(xt, w, b).to_a(Float32)
      end

      it "computes softmax of scaled, masked scores exactly as the composite does" do
        x = random(device, [2, 3, 4, 4], 7)
        mask = Tensor.causal_mask(4, 4, device)
        fused = x.softmax(scale: 0.35, mask: mask)
        fused.node.srcs.size.should eq 2
        fused.to_a(Float32).should eq softmax_composite(x * 0.35 + mask).to_a(Float32)
        xt = x.transpose
        xt.softmax(scale: 0.35).to_a(Float32).should eq softmax_composite(xt * 0.35).to_a(Float32)
      end

      it "computes softmax's gradient exactly as the composite does" do
        x, w = random(device, [2, 3, 5, 5], 8), random(device, [2, 3, 5, 5], 9)
        mask = random(device, [2, 3, 5, 5], 10)
        y = x.softmax(scale: 0.25, mask: mask)
        dx, dmask = Tensomite.grad((y * w).sum, [x, mask])
        dz = y * (w - (w * y).sum(-1, keepdims: true))
        dx.to_a(Float32).should eq (dz * 0.25).to_a(Float32)
        dmask.to_a(Float32).should eq dz.to_a(Float32)
        plain = x.softmax
        Tensomite.grad((plain * w).sum, [x]).first.to_a(Float32).should eq (plain * (w - (w * plain).sum(-1, keepdims: true))).to_a(Float32)
      end

      it "gives softmax of a row padded with -inf the same values" do
        x = random(device, [2, 5], 6)
        padded = Tensor.cat([x, Tensor.full([2, 3], -Float32::INFINITY, device: device)], 1)
        padded.softmax.slice(1, 0, 5).to_a(Float32).should eq x.softmax.to_a(Float32)
      end
    end
  end

  it "builds one node per op along the last axis, and composites elsewhere" do
    x = Tensor.from(f32(1, 2, 3, 4), [2, 2])
    x.softmax.node.op.should eq Op::Softmax
    x.log_softmax(1).node.op.should eq Op::LogSoftmax
    x.layernorm(Tensor.from(f32(1, 1), [2]), Tensor.from(f32(0, 0), [2])).node.op.should eq Op::LayerNorm
    x.softmax(0).node.op.should eq Op::Div
    x.softmax(scale: 2, mask: Tensor.from(f32(0, 1), [2])).node.op.should eq Op::Softmax
    x.softmax(0, scale: 2).node.op.should eq Op::Div
    Tensomite.grad(x.softmax(scale: 2).sum, [x]).first.node.op.should eq Op::SoftmaxGrad
    # softmax over axis 0 of [[1, 2], [3, 4]]: columns are [1, 3] and [2, 4]
    expect_close(x.softmax(0).to_a(Float32), [0.11920292, 0.11920292, 0.880797, 0.880797].map(&.to_f32))
  end

  it "differentiates each op" do
    check_gradients([[3, 5]], &.first.softmax)
    check_gradients([[3, 5], [5]]) { |(x, mask)| x.softmax(scale: 0.5, mask: mask) }
    check_gradients([[3, 5]], &.first.log_softmax)
    check_gradients([[2, 3, 6], [6], [6]]) { |(x, w, b)| x.layernorm(w, b) }
  end
end
