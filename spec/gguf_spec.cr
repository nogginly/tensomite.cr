require "./spec_helper"
require "json"

private def logits_fixture : {Array(Int32), Array(Array(Float32))}
  json = JSON.parse(File.read("#{__DIR__}/fixtures/tiny_llama/logits.json"))
  {json["prompt_ids"].as_a.map(&.as_i), json["logits"].as_a.map(&.as_a.map(&.as_f.to_f32))}
end

describe GGUF do
  dir = "#{__DIR__}/fixtures/tiny_llama"
  file = GGUF.open("#{dir}/model.gguf")

  it "reads metadata and tensor shapes, outermost first" do
    file.metadata["general.architecture"].should eq "llama"
    file.metadata["llama.block_count"].as_i.should eq 2
    file.metadata["llama.rope.freq_base"].as_f.should eq 10_000.0
    file.metadata["tokenizer.ggml.tokens"].as_a.size.should eq 400
    file.entries["token_embd.weight"].shape.should eq [400, 32]
    file.entries["blk.0.attn_k.weight"].shape.should eq [16, 32]
    file.entries["blk.0.attn_k.weight"].kind.should eq GGUF::Kind::F32
  end

  it "reads tensors on every device, widening F16" do
    safetensors = SafeTensors.open("#{dir}/model.safetensors")
    half = GGUF.open("#{dir}/model-f16.gguf")
    expected = safetensors["model.layers.0.self_attn.v_proj.weight"].to_a(Float32)
    DEVICES.each do |device|
      file["blk.0.attn_v.weight", device].to_a(Float32).should eq expected
      half["blk.0.attn_v.weight", device].to_a(Float32).should eq expected.map { |value| F16.new(value).to_f32 }
    end
  end

  it "rejects files that are not GGUF" do
    expect_raises(Error, /not a GGUF/) { GGUF.open("#{dir}/model.safetensors") }
  end
end

describe "BPE.from_gguf" do
  it "encodes as the tokenizer.json it was written from" do
    from_json = BPE.from_json("#{__DIR__}/fixtures/tokenizer/tokenizer.json")
    from_gguf = BPE.from_gguf(GGUF.open("#{__DIR__}/fixtures/tiny_llama/model.gguf"))
    from_gguf.specials.should eq from_json.specials
    JSON.parse(File.read("#{__DIR__}/fixtures/tokenizer/expected.json"))["texts"].as_a.each do |text|
      from_gguf.encode(text.as_s).should eq from_json.encode(text.as_s)
    end
  end
end

describe "Llama from GGUF" do
  dir = "#{__DIR__}/fixtures/tiny_llama"
  prompt, reference = logits_fixture

  (DEVICES + GPUS).each do |device|
    it "gives the safetensors model's logits, to the bit, at F32 on #{device.class}" do
      from_gguf = Llama.load("#{dir}/model.gguf", device)
      from_safetensors = Llama.load(dir, device)
      from_gguf.config.layers.should eq 2
      from_gguf.config.tied.should be_false
      expected = from_safetensors.forward(prompt, from_safetensors.new_cache).to_a(Float32)
      from_gguf.forward(prompt, from_gguf.new_cache).to_a(Float32).should eq expected
    end
  end

  it "stays close to PyTorch at F16" do
    model = Llama.load("#{dir}/model-f16.gguf")
    logits = model.forward(prompt, model.new_cache).to_a(Float32)
    width = model.config.vocab
    error = reference.each_with_index.max_of { |row, i| row.each_with_index.max_of { |value, j| (logits[i * width + j] - value).abs } }
    error.should be < 2e-2
  end
end

describe "M10: SmolLM2-135M-Instruct from GGUF" do
  path = ENV["TENSOMITE_SMOLLM2_GGUF"]? || ""
  if File.exists?(path)
    json = JSON.parse(File.read("#{__DIR__}/fixtures/smollm2/logits.json"))
    prompt = json["prompt_ids"].as_a.map(&.as_i)
    greedy = json["greedy"].as_a.map(&.as_i)
    file = GGUF.open(path)

    it "tokenises the ChatML prompt as Hugging Face does" do
      BPE.from_gguf(file).encode(json["prompt"].as_s).should eq prompt
    end

    it "answers greedily as PyTorch does from the BF16 weights" do
      model = Llama.load(file, BLAS.default)
      model.generate(prompt, 20).should eq greedy.first(20)
    end
  else
    pending "needs TENSOMITE_SMOLLM2_GGUF, an F16 GGUF of SmolLM2-135M-Instruct (see scripts/fetch_smollm2_gguf.sh)"
  end
end
