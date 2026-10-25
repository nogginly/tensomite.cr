require "./spec_helper"

private class TwoLayers < Tensomite::Module
  getter first : Linear
  getter second : Linear

  def initialize(rng : Random)
    @first = Linear.new(3, 4, rng: rng)
    @second = Linear.new(4, 2, bias: false, rng: rng)
  end

  def forward(x : Tensor) : Tensor
    @second.call(@first.call(x).gelu)
  end
end

private class Scaled < Tensomite::Module
  def initialize
    @norm = LayerNorm.new(4)
  end

  def forward(x : Tensor, scale : Float64) : Tensor
    @norm.call(x) * scale
  end
end

private class Stack < Tensomite::Module
  @blocks : Array(Scaled)
  @spare : Linear?

  def initialize(count : Int32)
    @blocks = Array.new(count) { Scaled.new }
    @spare = nil
  end

  def forward(x : Tensor) : Tensor
    @blocks.reduce(x) { |input, block| block.call(input, 0.5) }
  end
end

private class Tied < Tensomite::Module
  def initialize
    @embed = Embedding.new(5, 3)
    @shared = @embed.weight
  end

  def forward(ids : Tensor) : Tensor
    @embed.call(ids).matmul(@shared.value.transpose)
  end
end

describe Tensomite::Module do
  it "names parameters after instance variables, nested" do
    TwoLayers.new(Random.new(1)).named_params.keys.should eq ["first.weight", "first.bias", "second.weight"]
  end

  it "names the members of arrays by position, and skips nil parts" do
    Stack.new(2).named_params.keys.should eq ["blocks.0.norm.weight", "blocks.0.norm.bias", "blocks.1.norm.weight", "blocks.1.norm.bias"]
  end

  it "lists a parameter reached twice once, under its first name" do
    Tied.new.named_params.keys.should eq ["embed.weight"]
  end

  it "names each parameter's tensor, and keeps the name when the value is replaced" do
    model = TwoLayers.new(Random.new(1))
    param = model.named_params["first.weight"]
    param.value.name.should eq "first.weight"
    param.value = param.value * 2
    param.value.name.should eq "first.weight"
  end

  it "computes the same values as the tensor operations it wraps" do
    model = TwoLayers.new(Random.new(1))
    x = Tensor.from(random_values(6, 2), [2, 3])
    expected = x.matmul(model.first.weight.value) + model.first.bias.as(Parameter).value
    expected = expected.gelu.matmul(model.second.weight.value)
    model.call(x).to_a(Float32).should eq expected.to_a(Float32)
  end

  it "passes any arguments to forward" do
    Stack.new(1).call(Tensor.from(f32(1, 2, 3, 4), [1, 4])).shape.should eq [1, 4]
  end

  it "runs each child in a scope named after it, so diagrams group by module" do
    model = Stack.new(3)
    model.named_params
    text = model.call(Tensor.from(random_values(4, 3), [1, 4])).to_mermaid(model: model, depth: 1)
    text.should contain %(["blocks ×3<br/>2 ops · 8 params each"])
  end

  it "draws the same weights from the same generator" do
    a = TwoLayers.new(Random.new(7)).params.map(&.value.to_a(Float32))
    b = TwoLayers.new(Random.new(7)).params.map(&.value.to_a(Float32))
    a.should eq b
  end

  it "moves every parameter to another device" do
    model = TwoLayers.new(Random.new(1)).to(SimDiscrete.default)
    model.params.map(&.value.device).uniq!.should eq [SimDiscrete.default]
  end
end

describe Tensomite::Init do
  it "draws uniform values within the bound" do
    Init.uniform(0.5).tensor([1000], Random.new(1), CPU.default).to_a(Float32).all? { |value| value.abs <= 0.5 }.should be_true
  end

  it "draws normal values with about the standard deviation asked for" do
    values = Init.normal(0.02).tensor([10000], Random.new(1), CPU.default).to_a(Float32)
    mean = values.sum / values.size
    std = Math.sqrt(values.sum { |value| (value - mean) ** 2 } / values.size)
    std.should be_close(0.02, 0.001)
  end

  it "fills constants" do
    Init.constant(3).tensor([2], Random.new(1), CPU.default).to_a(Float32).should eq f32(3, 3)
  end
end

describe "built-in modules" do
  it "flags weights for decay, and biases and norms not" do
    TwoLayers.new(Random.new(1)).params.map(&.decay?).should eq [true, false, true]
    LayerNorm.new(3).params.map(&.decay?).should eq [false, false]
    Embedding.new(4, 3).params.map(&.decay?).should eq [true]
  end

  it "embeds ids of any shape" do
    embedding = Embedding.new(10, 3, rng: Random.new(1))
    ids = Tensor.from([1, 2, 3, 4, 5, 6], [2, 3])
    out = embedding.call(ids)
    out.shape.should eq [2, 3, 3]
    out.to_a(Float32)[0, 3].should eq embedding.weight.value.to_a(Float32)[3, 3]
  end

  it "chains layers in a Sequential, naming steps by kind and position" do
    mlp = Sequential.new(Linear.new(2, 3), Activation.gelu, Linear.new(3, 1))
    mlp.named_params.keys.should eq ["steps.0.weight", "steps.0.bias", "steps.2.weight", "steps.2.bias"]
    x = Tensor.from(f32(1, 2), [1, 2])
    first, last = mlp.steps[0].as(Linear), mlp.steps[2].as(Linear)
    expected = (x.matmul(first.weight.value) + first.bias.as(Parameter).value).gelu
    expected = expected.matmul(last.weight.value) + last.bias.as(Parameter).value
    mlp.call(x).to_a(Float32).should eq expected.to_a(Float32)
  end
end

describe "optimizers given parameters" do
  it "update the parameters in place and return the loss" do
    model = TwoLayers.new(Random.new(1))
    optimizer = AdamW.new(model.params, lr: 1e-2)
    x = Tensor.from(random_values(12, 3), [4, 3])
    losses = Array.new(20) { optimizer.step((model.call(x) * model.call(x)).mean) }
    losses.last.should be < losses.first
  end

  it "give the same values as stepping tensors" do
    model = TwoLayers.new(Random.new(1))
    tensors = model.params.map(&.value)
    x = Tensor.from(random_values(12, 3), [4, 3])
    by_tensors = AdamW.new(lr: 1e-2, weight_decay: 0.0).step(tensors, Tensomite.grad(model.call(x).sum, tensors))
    AdamW.new(model.params, lr: 1e-2, weight_decay: 0.0).step(model.call(x).sum)
    model.params.map(&.value.to_a(Float32)).should eq by_tensors.map(&.to_a(Float32))
  end

  it "decay only the parameters flagged for it" do
    model = LayerNorm.new(2)
    before = model.params.map(&.value.to_a(Float32))
    x = Tensor.from(f32(1, 2), [1, 2])
    AdamW.new(model.params, lr: 0.1, weight_decay: 0.5).step((model.call(x) * 0).sum)
    model.params.map(&.value.to_a(Float32)).should eq before

    linear = Linear.new(2, 1, bias: false, rng: Random.new(1))
    weight = linear.weight.value.to_a(Float32)
    AdamW.new(linear.params, lr: 0.1, weight_decay: 0.5).step((linear.call(x) * 0).sum)
    linear.weight.value.to_a(Float32).should eq weight.map { |value| value * (1 - 0.1 * 0.5).to_f32 }
  end

  it "refuse step(loss) when created without parameters" do
    expect_raises(Error, /parameters/) { SGD.new(lr: 0.1).step(Tensor.from(f32(1), [1]).sum) }
  end
end

describe "saving and loading modules" do
  it "loads a saved model's parameters by name" do
    path = File.tempname("tensomite", ".safetensors")
    trained = TwoLayers.new(Random.new(1))
    trained.save(path)
    SafeTensors.open(path).names.should eq ["first.weight", "first.bias", "second.weight"]
    fresh = TwoLayers.new(Random.new(2)).load(path)
    fresh.params.map(&.value.to_a(Float32)).should eq trained.params.map(&.value.to_a(Float32))
    fresh.named_params["first.weight"].value.name.should eq "first.weight"
  end

  it "loads onto the device each parameter is on" do
    path = File.tempname("tensomite", ".safetensors")
    TwoLayers.new(Random.new(1)).save(path)
    model = TwoLayers.new(Random.new(2)).to(SimDiscrete.default).load(path)
    model.params.map(&.value.device).uniq!.should eq [SimDiscrete.default]
  end

  it "refuses missing, extra and misshapen tensors when strict, and skips the first two otherwise" do
    path = File.tempname("tensomite", ".safetensors")
    SafeTensors.write(path, {"first.weight" => Tensor.zeros([3, 4]), "other" => Tensor.zeros([1])})
    expect_raises(Error, /lacks first.bias, second.weight/) { TwoLayers.new(Random.new(1)).load(path) }
    TwoLayers.new(Random.new(1)).load(path, strict: false).named_params["first.weight"].value.to_a(Float32).should eq Array.new(12, 0f32)

    SafeTensors.write(path, {"first.weight" => Tensor.zeros([4, 3]), "first.bias" => Tensor.zeros([4]), "second.weight" => Tensor.zeros([4, 2])})
    expect_raises(ShapeError, /first.weight/) { TwoLayers.new(Random.new(1)).load(path) }

    SafeTensors.write(path, {"first.weight" => Tensor.zeros([3, 4]), "first.bias" => Tensor.zeros([4]), "second.weight" => Tensor.zeros([4, 2]), "other" => Tensor.zeros([1])})
    expect_raises(Error, /has other/) { TwoLayers.new(Random.new(1)).load(path) }
  end
end
