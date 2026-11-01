require "./spec_helper"

# A CPU that counts how often it is waited for.
private class CountingCPU < CPU
  getter syncs = 0

  def sync : Nil
    @syncs += 1
  end
end

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

  it "gives the bits of its composite of primitive ops, step after step" do
    params = [Parameter.new(Tensor.from(random_values(12, 1), [3, 4])), Parameter.new(Tensor.from(random_values(4, 2), [4]), decay: false)]
    optimizer = AdamW.new(params, lr: 0.05, beta2: 0.99, weight_decay: 0.1)
    values = params.map(&.value)
    m = [] of Tensor
    v = [] of Tensor
    (1..4).each do |step|
      optimizer.lr = 0.05 / step
      grads = [Tensor.from(random_values(12, 10 + step), [3, 4]), Tensor.from(random_values(4, 20 + step), [4])]
      loss = (params[0].value * grads[0]).sum + (params[1].value * grads[1]).sum
      optimizer.step(loss)
      values = values.map_with_index do |value, i|
        m_i = grads[i] * 0.1
        v_i = grads[i] * grads[i] * 0.01
        m_i, v_i = m[i] * 0.9 + m_i, v[i] * 0.99 + v_i if step > 1
        m.size > i ? (m[i] = m_i) : m << m_i
        v.size > i ? (v[i] = v_i) : v << v_i
        decayed = i == 0 ? value * (1 - 0.05 / step * 0.1) : value
        (decayed - m_i / (1 - 0.9 ** step) / ((v_i / (1 - 0.99 ** step)).sqrt + 1e-8) * (0.05 / step)).realize
      end
      params.zip(values) { |param, value| param.value.to_a(Float32).should eq value.to_a(Float32) }
    end
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

describe Optimizer do
  it "waits for the device once per step, with the loss, state and updates" do
    device = CountingCPU.new
    params = [Parameter.new(Tensor.from(f32(1, 2), [2], device)), Parameter.new(Tensor.from(f32(3), [1], device))]
    optimizer = AdamW.new(params, lr: 0.1)
    2.times do
      before = device.syncs
      loss = (params[0].value * params[0].value).sum + (params[1].value * 2).sum
      optimizer.step(loss).should be > 0
      (device.syncs - before).should eq 1
    end
  end
end
