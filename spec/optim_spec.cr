require "./spec_helper"

describe SGD do
  it "steps against the gradient" do
    p = [Tensor.from(f32(1, 2), [2])]
    p = SGD.new(lr: 0.5).step(p, [Tensor.from(f32(2, -2), [2])])
    p.first.to_a(Float32).should eq f32(0, 3)
  end

  it "accumulates momentum" do
    optimizer = SGD.new(lr: 1.0, momentum: 0.5)
    p = [Tensor.from(f32(0), [1])]
    g = [Tensor.from(f32(1), [1])]
    p = optimizer.step(p, g)
    p = optimizer.step(p, g)
    p.first.to_a(Float32).should eq f32(-2.5)
  end

  it "rejects mismatched gradients" do
    expect_raises(ShapeError) { SGD.new(lr: 0.1).step([Tensor.from(f32(1), [1])], [Tensor.from(f32(1, 2), [2])]) }
  end
end

describe AdamW do
  it "matches the reference update for two steps" do
    optimizer = AdamW.new(lr: 0.1, weight_decay: 0.1)
    p = [Tensor.from(f32(1, -1), [2])]
    p = optimizer.step(p, [Tensor.from(f32(0.5, 0.2), [2])])
    p = optimizer.step(p, [Tensor.from(f32(-0.3, 0.4), [2])])
    # Reference, in double precision, from the formulas in the docs.
    expected = [{1.0, [0.5, -0.3]}, {-1.0, [0.2, 0.4]}].map do |(start, grads)|
      x = start
      m = v = 0.0
      grads.each_with_index do |grad, i|
        t = i + 1
        x -= 0.1 * 0.1 * x
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad * grad
        x -= 0.1 * (m / (1 - 0.9 ** t)) / (Math.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
      end
      x.to_f32
    end
    expect_close(p.first.to_a(Float32), expected, 1e-5)
  end

  it "minimises a quadratic" do
    optimizer = AdamW.new(lr: 0.1, weight_decay: 0.0)
    target = Tensor.from(f32(3, -2), [2])
    p = [Tensor.from(f32(0, 0), [2])]
    300.times do
      diff = p.first - target
      p = optimizer.step(p, Tensomite.grad((diff * diff).sum, p))
    end
    expect_close(p.first.to_a(Float32), f32(3, -2), 1e-2)
  end
end
