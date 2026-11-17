require "./spec_helper"

private def scratch(name : String) : String
  File.join(Dir.tempdir, "tensomite-#{Process.pid}-#{name}")
end

private def tiny_llama(seed : Int32) : Llama
  config = Llama::Config.new(layers: 2, heads: 4, kv_heads: 2, width: 64, hidden: 96, head_width: 16, context: 32, vocab: 64)
  Llama.new(config, Builder.new(rng: Random.new(seed)))
end

describe "GGUF.write" do
  it "writes tensors that read back byte for byte, with their metadata" do
    path = scratch("round.gguf")
    values = random_values(3 * 64, 1)
    tensors = {
      "plain" => Tensor.from(values, [3, 64]),
      "q8"    => Tensor.from(values, [3, 64]).quantize(DType::Q8_0),
      "q4"    => Tensor.from(values, [3, 64]).quantize(DType::Q4_0),
      "q41"   => Tensor.from(values, [3, 64]).quantize(DType::Q4_1),
      "row"   => Tensor.from(values[0, 5], [5]),
    }
    metadata = {
      "general.name" => JSON::Any.new("tiny"), "count" => JSON::Any.new(7_i64), "big" => JSON::Any.new(1_i64 << 40),
      "ratio" => JSON::Any.new(0.25), "flag" => JSON::Any.new(true),
      "words" => JSON::Any.new(["a", "b"].map { |word| JSON::Any.new(word) }),
    }
    GGUF.write(path, tensors, metadata)
    file = GGUF.open(path)
    file.names.should eq tensors.keys
    tensors.each do |name, tensor|
      file.entries[name].shape.should eq tensor.shape
      file.entries[name].kind.dtype.should eq tensor.dtype
      file[name].bytes.should eq tensor.bytes
    end
    file.metadata.should eq metadata
  ensure
    File.delete?(path) if path
  end

  it "raises for an I32 tensor" do
    expect_raises(DTypeError, /not I32/) { GGUF.write(scratch("ids.gguf"), {"ids" => Tensor.from([1, 2], [2])}) }
  end
end

describe "Module#save and #load with GGUF" do
  it "loads a quantised model into one built in F32, with the same logits" do
    path = scratch("model.gguf")
    model = tiny_llama(1).quantize(DType::Q8_0, embeddings: true)
    model.save(path, {"note" => "tiny"})
    GGUF.open(path).metadata["note"].as_s.should eq "tiny"
    loaded = tiny_llama(2).load(path)
    loaded.named_params.each { |name, param| param.value.dtype.should eq model.named_params[name].value.dtype }
    ids = [3, 1, 4, 1, 5]
    loaded.forward(ids, loaded.new_cache).to_a(Float32).should eq model.forward(ids, model.new_cache).to_a(Float32)
  ensure
    File.delete?(path) if path
  end

  it "is about a quarter of the size in Q8_0, a seventh in Q4_0" do
    plain, q8, q4 = scratch("plain.gguf"), scratch("q8.gguf"), scratch("q4.gguf")
    tiny_llama(1).save(plain)
    tiny_llama(1).quantize(DType::Q8_0, embeddings: true).save(q8)
    tiny_llama(1).quantize(DType::Q4_0, embeddings: true).save(q4)
    (File.size(q8) / File.size(plain)).should be < 0.3
    (File.size(q4) / File.size(plain)).should be < 0.17
  ensure
    [plain, q8, q4].each { |path| File.delete?(path) if path }
  end

  it "raises for a tensor whose shape differs from its parameter's" do
    path = scratch("wrong.gguf")
    GGUF.write(path, tiny_llama(1).named_params.transform_values(&.value).merge({"model.norm.weight" => Tensor.zeros([3])}))
    expect_raises(ShapeError, /model.norm.weight/) { tiny_llama(2).load(path) }
  ensure
    File.delete?(path) if path
  end
end

describe "GPT2#save_gguf and GPT2.load of a GGUF file" do
  it "reads a quantised GPT-2 back, heads included, with the same logits" do
    path = scratch("gpt2.gguf")
    model = GPT2.load("#{__DIR__}/fixtures/tiny_gpt2/model.safetensors", heads: 4).quantize(DType::Q8_0, embeddings: true)
    model.save_gguf(path)
    loaded = GPT2.load(path)
    loaded.config.should eq model.config
    ids = [1, 2, 3, 4]
    loaded.forward(ids, loaded.new_cache).to_a(Float32).should eq model.forward(ids, model.new_cache).to_a(Float32)
  ensure
    File.delete?(path) if path
  end
end
