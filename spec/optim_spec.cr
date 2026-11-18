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

describe "Optimizer state" do
  it "restores AdamW's m, v and step count, so the next step has the same bits" do
    run = ->(optimizer : AdamW, p : Array(Tensor), step : Int32) do
      g = [Tensor.from(random_values(6, 30 + step), [2, 3]), Tensor.from(random_values(3, 40 + step), [3])]
      optimizer.step(p, g)
    end
    start = [Tensor.from(random_values(6, 1), [2, 3]), Tensor.from(random_values(3, 2), [3])]
    straight = AdamW.new(lr: 0.01)
    p = start
    3.times { |i| p = run.call(straight, p, i) }

    first = AdamW.new(lr: 0.01)
    q = start
    2.times { |i| q = run.call(first, q, i) }
    path = File.tempname("optimizer", ".safetensors")
    SafeTensors.write(path, first.state)
    first.state.keys.sort!.should eq ["m.0", "m.1", "v.0", "v.1"]
    resumed = AdamW.new(lr: 0.01).restore(SafeTensors.open(path), steps: first.steps)
    resumed.steps.should eq 2
    q = run.call(resumed, q, 2)
    q.map(&.to_a(Float32)).should eq p.map(&.to_a(Float32))
  ensure
    File.delete(path) if path && File.exists?(path)
  end

  it "rejects state it does not keep" do
    path = File.tempname("optimizer", ".safetensors")
    SafeTensors.write(path, {"velocity.0" => Tensor.from(f32(1), [1])})
    expect_raises(Error, /no state named velocity.0/) { AdamW.new.restore(SafeTensors.open(path), steps: 1) }
  ensure
    File.delete(path) if path && File.exists?(path)
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

# Returns fresh parameters with the same values as *values*, each decaying as *decay* says.
private def params_of(values : Array(Array(Float32)), decay : Array(Bool)) : Array(Parameter)
  values.map_with_index { |value, i| Parameter.new(Tensor.from(value, [value.size]), decay: decay[i]) }
end

# Takes *steps* steps of *optimizers*, each over its own parameters, on the
# same quadratic loss, and returns every parameter's values.
private def train(optimizers : Array({Optimizer, Array(Parameter)}), steps : Int32, &schedule : Int32 ->) : Array(Array(Float32))
  steps.times do |step|
    schedule.call(step)
    optimizers.each do |(optimizer, params)|
      loss = params.map { |param| (param.value * param.value * param.value).sum }.reduce { |a, b| a + b }
      optimizer.step(loss)
    end
  end
  optimizers.flat_map { |(_, params)| params.map(&.value.to_a(Float32)) }
end

describe AdamW::Group do
  values = [random_values(4, 1), random_values(3, 2), random_values(5, 3)]
  decay = [true, false, true]

  it "gives each group the bits of a separate AdamW with its settings" do
    grouped = params_of(values, decay)
    a = AdamW::Group.new(grouped[0, 2], lr: 0.05)
    b = AdamW::Group.new(grouped[2, 1], lr: 0.02, beta1: 0.8, beta2: 0.95, weight_decay: 0.1)
    optimizer = AdamW.new([a, b], weight_decay: 0.01)
    got = train([{optimizer.as(Optimizer), grouped}], 4) { }

    first, second = params_of(values, decay)[0, 2], params_of(values, decay)[2, 1]
    apart = [{AdamW.new(first, lr: 0.05, weight_decay: 0.01).as(Optimizer), first},
             {AdamW.new(second, lr: 0.02, beta1: 0.8, beta2: 0.95, weight_decay: 0.1).as(Optimizer), second}]
    train(apart, 4) { }.should eq got
  end

  it "lets a schedule scale each group's rate from its initial one" do
    params = params_of(values, decay)
    a, b = AdamW::Group.new(params[0, 1], lr: 0.1), AdamW::Group.new(params[1, 2])
    optimizer = AdamW.new([a, b], lr: 0.01)
    b.initial_lr.should eq 0.01
    optimizer.groups.each { |group| group.lr = group.initial_lr * 0.5 }
    {a.lr, b.lr, a.initial_lr}.should eq({0.05, 0.005, 0.1})
    optimizer.lr = 0.2
    optimizer.groups.map(&.lr).should eq [0.2, 0.2]
  end

  it "keeps state named across the groups, as one list of params is" do
    params = params_of(values, decay)
    optimizer = AdamW.new([AdamW::Group.new(params[0, 1]), AdamW::Group.new(params[1, 2], lr: 0.1)])
    train([{optimizer.as(Optimizer), params}], 1) { }
    optimizer.state.keys.sort!.should eq ["m.0", "m.1", "m.2", "v.0", "v.1", "v.2"]
    single = params_of(values, decay)
    one = AdamW.new(single)
    train([{one.as(Optimizer), single}], 1) { }
    one.state.keys.sort!.should eq optimizer.state.keys.sort!
  end
end

# An optimizer that updates its params two at a time: each pair moves against
# the mean of its gradients, with a running sum of steps as its state.
private class PairOptimizer < Optimizer
  @moves = {} of Int32 => Tensor

  def initialize(params : Array(Parameter))
    @slots = params
  end

  protected def tables : Hash(String, Hash(Int32, Tensor))
    {"moves" => @moves}
  end

  protected def updates(params : Array(Tensor), grads : Array(Tensor)) : Array(Tensor)
    params.each_slice(2).with_index.flat_map do |(pair, n)|
      i = n * 2
      mean = (grads[i] + grads[i + 1]) * 0.5
      move = keep(@moves, i, @moves[i]?.try { |prev| prev + mean } || mean)
      pair.map { |param| param - move * 0.1 }
    end.to_a
  end

  protected def update(index : Int32, param : Tensor, grad : Tensor) : Tensor
    raise "a pair optimizer updates params in pairs"
  end
end

describe "Optimizer#updates" do
  it "lets an optimizer update params together, with its state saved and restored" do
    make = -> { [Parameter.new(Tensor.from(f32(1, 2), [2])), Parameter.new(Tensor.from(f32(3, -1), [2]))] }
    loss_of = ->(params : Array(Parameter)) { (params[0].value * params[0].value).sum + (params[1].value * 3).sum }
    straight = make.call
    optimizer = PairOptimizer.new(straight)
    3.times { optimizer.step(loss_of.call(straight)) }

    resumed_params = make.call
    first = PairOptimizer.new(resumed_params)
    2.times { first.step(loss_of.call(resumed_params)) }
    first.state.keys.should eq ["moves.0"]
    path = File.tempname("pairs", ".safetensors")
    SafeTensors.write(path, first.state)
    second = PairOptimizer.new(resumed_params).restore(SafeTensors.open(path), steps: first.steps)
    second.step(loss_of.call(resumed_params))
    resumed_params.map(&.value.to_a(Float32)).should eq straight.map(&.value.to_a(Float32))
  ensure
    File.delete(path) if path && File.exists?(path)
  end
end
