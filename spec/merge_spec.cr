require "./spec_helper"

private DIR = "#{__DIR__}/fixtures/tiny_llama"

# Returns *model* with every weight replaced by a copy of itself, so that no
# projection holds a share of a joined weight any more.
private def unmerged(model : Llama) : Llama
  model.params.each { |param| param.value = param.value.detach }
  model
end

private def logits(model : Llama, ids : Array(Int32)) : Array(Float32)
  model.forward(ids, model.new_cache).to_a(Float32)
end

describe "Llama's joined projections" do
  ids = [1, 85, 91, 264, 384, 201, 36]

  (DEVICES + GPUS).each do |device|
    {"q8_0", "q4_0"}.each do |name|
      it "give the separate products' bits at #{name} on #{device.class}" do
        joined = Llama.load("#{DIR}/model-#{name}.gguf", device)
        separate = unmerged(Llama.load("#{DIR}/model-#{name}.gguf", device))
        logits(joined, ids).should eq logits(separate, ids)
      end
    end

    if device.shape_invariant?
      it "give the separate products' bits at F32 on #{device.class}" do
        logits(Llama.load(DIR, device), ids).should eq logits(unmerged(Llama.load(DIR, device)), ids)
      end
    end
  end

  it "hold each weight once, as views of the joined one" do
    model = Llama.load("#{DIR}/model-q4_0.gguf")
    q, k, v = %w[q_proj k_proj v_proj].map { |part| model.named_params["model.layers.0.self_attn.#{part}.weight"].value }
    q.node.should be k.node
    k.node.should be v.node
    q.shape.should eq [32, 32]
    k.shape.should eq [16, 32]
    gate = model.named_params["model.layers.0.mlp.gate_proj.weight"].value
    gate.node.should be model.named_params["model.layers.0.mlp.up_proj.weight"].value.node
  end

  it "join again after quantising, and stay joined at F32" do
    model = Llama.load(DIR)
    model.named_params["model.layers.1.self_attn.q_proj.weight"].value.node
      .should be model.named_params["model.layers.1.self_attn.v_proj.weight"].value.node
    model.quantize(DType::Q8_0)
    model.named_params["model.layers.1.mlp.gate_proj.weight"].value.dtype.should eq DType::Q8_0
    model.named_params["model.layers.1.mlp.gate_proj.weight"].value.node
      .should be model.named_params["model.layers.1.mlp.up_proj.weight"].value.node
  end

  it "fall back to separate products once a weight is replaced" do
    model = Llama.load(DIR)
    before = logits(model, ids)
    param = model.named_params["model.layers.0.self_attn.k_proj.weight"]
    param.value = param.value.detach
    logits(model, ids).should eq before
  end

  it "skip parts whose formats differ" do
    a = Linear.new(32, 8, bias: false)
    b = Linear.new(32, 8, bias: false)
    a.quantize(DType::Q8_0)
    Linear.merge([a, b]).should be_nil
    Linear.merge([Linear.new(32, 8), Linear.new(32, 8)]).should be_nil
  end
end
