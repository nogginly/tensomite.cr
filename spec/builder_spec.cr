require "./spec_helper"

# The tutorial's and the Shakespeare models three ways: as plain modules, with a
# Builder (candidate A) and with `parts` (candidate B).

private class Multiplier < Tensomite::Module
  def initialize(rng : Random)
    @hidden1 = Linear.new(2, 64, rng: rng)
    @hidden2 = Linear.new(64, 64, rng: rng)
    @output = Linear.new(64, 1, rng: rng)
  end

  def forward(x : Tensor) : Tensor
    @output.call(@hidden2.call(@hidden1.call(x).gelu).gelu)
  end
end

private class MultiplierA < Tensomite::Module
  def initialize(build : Builder = Builder.new)
    @hidden1 = build.linear(2, 64)
    @hidden2 = build.linear(64, 64)
    @output = build.linear(64, 1)
  end

  def forward(x : Tensor) : Tensor
    @output.call(@hidden2.call(@hidden1.call(x).gelu).gelu)
  end
end

private class MultiplierB < Tensomite::Module
  parts do
    hidden1 = linear(2, 64)
    hidden2 = linear(64, 64)
    output = linear(64, 1)
  end

  def forward(x : Tensor) : Tensor
    output(hidden2(hidden1(x).gelu).gelu)
  end
end

private class Block < Tensomite::Module
  def initialize(width : Int32, heads : Int32, residual : Init, rng : Random, device : Device)
    @ln1 = LayerNorm.new(width, device: device)
    @attention = CausalSelfAttention.new(width, heads, bias: false, proj_init: residual, rng: rng, device: device)
    @ln2 = LayerNorm.new(width, device: device)
    @mlp = Sequential.new(
      Linear.new(width, 4 * width, bias: false, init: Init.normal(0.02), rng: rng, device: device),
      Activation.gelu,
      Linear.new(4 * width, width, bias: false, init: residual, rng: rng, device: device))
  end

  def forward(x : Tensor, mask : Tensor) : Tensor
    x = x + @attention.call(@ln1.call(x), mask)
    x + @mlp.call(@ln2.call(x))
  end
end

private class CharGPT < Tensomite::Module
  getter width : Int32
  @blocks : Array(Block)

  def initialize(layers : Int32, heads : Int32, @width : Int32, context : Int32, vocab : Int32, device : Device, seed : Int32)
    rng = Random.new(seed)
    @tokens = Embedding.new(vocab, width, rng: rng, device: device)
    @positions = Parameter.new(Init.normal(0.02).tensor([context, width], rng, device))
    residual = Init.normal(0.02 / Math.sqrt(2 * layers))
    @blocks = Array.new(layers) { Block.new(width, heads, residual, rng, device) }
    @ln = LayerNorm.new(width, device: device)
  end

  def forward(ids : Tensor) : Tensor
    batch, length = ids.shape
    x = Tensomite.scope("embedding") do
      (@tokens.call(ids) + @positions.value.slice(0, 0, length)).reshape([batch * length, width])
    end
    mask = Tensor.causal_mask(length, length, ids.device)
    @blocks.each { |block| x = block.call(x, mask) }
    Tensomite.scope("head") { @ln.call(x).matmul(@tokens.weight.value.transpose) }
  end
end

private class BlockA < Tensomite::Module
  def initialize(width : Int32, heads : Int32, residual : Init, build : Builder)
    @ln1 = build.layernorm(width)
    @attention = build.attention(width, heads, proj_init: residual)
    @ln2 = build.layernorm(width)
    @mlp = Sequential.build(build) do
      linear width, 4 * width
      gelu
      linear 4 * width, width, init: residual
    end
  end

  def forward(x : Tensor, mask : Tensor) : Tensor
    x = x + @attention.call(@ln1.call(x), mask)
    x + @mlp.call(@ln2.call(x))
  end
end

private class CharGPTA < Tensomite::Module
  getter width : Int32
  @blocks : Array(BlockA)

  def initialize(layers : Int32, heads : Int32, @width : Int32, context : Int32, vocab : Int32, build : Builder)
    @tokens = build.embedding(vocab, width)
    @positions = build.parameter([context, width])
    residual = Init.normal(0.02 / Math.sqrt(2 * layers))
    @blocks = Array.new(layers) { BlockA.new(width, heads, residual, build) }
    @ln = build.layernorm(width)
  end

  def forward(ids : Tensor) : Tensor
    batch, length = ids.shape
    x = Tensomite.scope("embedding") do
      (@tokens.call(ids) + @positions.value.slice(0, 0, length)).reshape([batch * length, width])
    end
    mask = Tensor.causal_mask(length, length, ids.device)
    @blocks.each { |block| x = block.call(x, mask) }
    Tensomite.scope("head") { @ln.call(x).matmul(@tokens.weight.value.transpose) }
  end
end

private class BlockB < Tensomite::Module
  parts(width : Int32, heads : Int32, residual : Init) do
    ln1 = layernorm(width)
    attention = attention(width, heads, proj_init: residual)
    ln2 = layernorm(width)
    mlp = sequential do
      linear width, 4 * width
      gelu
      linear 4 * width, width, init: residual
    end
  end

  def forward(x : Tensor, mask : Tensor) : Tensor
    x = x + attention(ln1(x), mask)
    x + mlp(ln2(x))
  end
end

private class CharGPTB < Tensomite::Module
  parts(layers : Int32, heads : Int32, width : Int32, context : Int32, vocab : Int32) do
    tokens = embedding(vocab, width)
    positions = parameter([context, width])
    blocks : Array(BlockB) = Array.new(layers) { BlockB.new(width, heads, Init.normal(0.02 / Math.sqrt(2 * layers)), build) }
    ln = layernorm(width)
  end

  def forward(ids : Tensor) : Tensor
    batch, length = ids.shape
    x = Tensomite.scope("embedding") do
      (tokens(ids) + positions.value.slice(0, 0, length)).reshape([batch * length, width])
    end
    mask = Tensor.causal_mask(length, length, ids.device)
    blocks.each { |block| x = block.call(x, mask) }
    Tensomite.scope("head") { ln(x).matmul(tokens.weight.value.transpose) }
  end
end

# Returns each model's parameter names and values after one AdamW step on *loss*.
private def trained(model : Tensomite::Module, & : -> Tensor) : Hash(String, Array(Float32))
  AdamW.new(model.params, lr: 1e-2).step(yield)
  model.named_params.transform_values(&.value.to_a(Float32))
end

describe Builder do
  it "builds the tutorial's model as plain modules do, either way" do
    x = Tensor.from(random_values(8, 1), [4, 2])
    expected = Multiplier.new(Random.new(1))
    a = MultiplierA.new(Builder.new(rng: Random.new(1)))
    b = MultiplierB.new(Builder.new(rng: Random.new(1)))
    want = trained(expected) { expected.call(x).sum }
    trained(a) { a.call(x).sum }.should eq want
    trained(b) { b.call(x).sum }.should eq want
  end

  it "builds the Shakespeare model as plain modules do, either way" do
    ids = Tensor.from(Array.new(2 * 8) { |i| i * 7 % 11 }, [2, 8])
    targets = Tensor.from(Array.new(2 * 8) { |i| i * 5 % 11 }, [16])
    expected = CharGPT.new(2, 2, 16, 8, 11, CPU.default, 3)
    build = Builder.new(rng: Random.new(3), bias: false, init: Init.normal(0.02))
    a = CharGPTA.new(2, 2, 16, 8, 11, build)
    b = CharGPTB.new(2, 2, 16, 8, 11, build.with(rng: Random.new(3)))
    want = trained(expected) { expected.call(ids).cross_entropy(targets) }
    trained(a) { a.call(ids).cross_entropy(targets) }.should eq want
    trained(b) { b.call(ids).cross_entropy(targets) }.should eq want
  end

  it "makes parts on its device, with its defaults unless told otherwise" do
    build = Builder.new(device: SimDiscrete.default, bias: false)
    build.linear(2, 3).bias.should be_nil
    build.linear(2, 3, bias: true).bias.should_not be_nil
    build.with(bias: true).linear(2, 3).bias.should_not be_nil
    build.layernorm(3).params.map(&.value.device).uniq!.should eq [SimDiscrete.default]
  end

  it "chains the layers a sequential block adds" do
    mlp = Builder.new.sequential do
      linear 2, 3
      gelu
      add Activation.tanh
    end
    mlp.steps.map(&.kind).should eq ["linear", "gelu", "tanh"]
  end
end

describe "Module.parts" do
  it "makes hyperparameters getters and parts callable by name" do
    block = BlockB.new(8, 2, Init.normal(0.01))
    {block.width, block.heads}.should eq({8, 2})
    block.ln1.should be_a LayerNorm
    x = Tensor.from(random_values(8, 2), [1, 8])
    block.ln1(x).to_a(Float32).should eq block.ln1.call(x).to_a(Float32)
  end

  it "runs each part in a scope named after it" do
    model = MultiplierB.new
    text = model.call(Tensor.zeros([1, 2])).to_mermaid(model: model, depth: 1)
    text.should contain "hidden1"
    text.should contain "output"
  end
end
