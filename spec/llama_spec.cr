require "./spec_helper"
require "json"

# Reference outputs from scripts/make_fixtures.py: PyTorch's Llama run on the
# same weights and Hugging Face's tokenizers on the same tokeniser.
private record LlamaExpected, prompt : String, prompt_ids : Array(Int32), logit_rows : Array(Int32),
  logits : Array(Array(Float32)), greedy : Array(Int32) do
  def self.read(path : String) : LlamaExpected
    json = JSON.parse(File.read(path))
    ints = ->(any : JSON::Any) { any.as_a.map(&.as_i) }
    new(json["prompt"].as_s, ints.call(json["prompt_ids"]), ints.call(json["logit_rows"]),
      json["logits"].as_a.map { |row| row.as_a.map(&.as_f.to_f32) }, ints.call(json["greedy"]))
  end

  # Returns the largest absolute difference between the expected rows and *logits*.
  def error(logits : Tensor) : Float32
    width = logits.shape[1]
    values = logits.to_a(Float32)
    logit_rows.zip(logits_expected).max_of do |row, want|
      want.each_with_index.max_of { |value, col| (values[row * width + col] - value).abs }
    end
  end

  private def logits_expected : Array(Array(Float32))
    logits
  end
end

# Runs *ids* one position at a time and returns the logits of every position.
private def decode_one_by_one(model : Llama, ids : Array(Int32)) : Array(Float32)
  cache = model.new_cache(ids.size)
  ids.flat_map { |id| model.forward([id], cache).to_a(Float32) }
end

describe "Llama on the tiny reference model" do
  dir = "#{__DIR__}/fixtures/tiny_llama"
  expected = LlamaExpected.read("#{dir}/logits.json")
  bpe = BPE.from_json("#{__DIR__}/fixtures/tokenizer/tokenizer.json")

  it "reads its configuration from config.json" do
    Llama.load(dir).config.should eq Llama::Config.new(
      layers: 2, heads: 4, kv_heads: 2, width: 32, hidden: 64, head_width: 8, context: 64, vocab: 400,
      rope_theta: 10_000.0, eps: 1e-5, tied: false)
  end

  it "tokenises the ChatML prompt like the reference" do
    bpe.encode(expected.prompt).should eq expected.prompt_ids
  end

  (DEVICES + GPUS).each do |device|
    describe "on #{device.class}" do
      model = Llama.load(dir, device)

      it "matches the reference logits within 1e-4" do
        expected.error(model.forward(expected.prompt_ids, model.new_cache)).should be < 1e-4
      end

      it "decodes greedily like the reference" do
        model.generate(expected.prompt_ids, expected.greedy.size).should eq expected.greedy
      end

      if device.shape_invariant?
        it "gives bit-identical logits in prefill and in one-by-one decode" do
          prefill = model.forward(expected.prompt_ids, model.new_cache).to_a(Float32)
          decode_one_by_one(model, expected.prompt_ids).should eq prefill
        end

        it "gives a batch's logits, row by row, as each sequence's own prefill" do
          ids = expected.prompt_ids.first(8)
          batch = model.call(Tensor.from(ids + ids.reverse, [2, 8], device)).to_a(Float32)
          batch.should eq [ids, ids.reverse].flat_map { |row| model.forward(row, model.new_cache).to_a(Float32) }
        end
      end
    end
  end

  it "names its parameters as the weights file does" do
    names = Llama.load(dir).named_params.keys
    names.should eq SafeTensors.open("#{dir}/model.safetensors").names.sort_by { |name| names.index(name) || -1 }
    names.first.should eq "model.embed_tokens.weight"
    names.should contain "lm_head.weight"
  end

  it "stops generating after a stop token" do
    model = Llama.load(dir)
    stop = expected.greedy[3]
    model.generate(expected.prompt_ids, 20, stop: [stop]).should eq expected.greedy[0..3]
  end

  it "trains: the loss's gradient reaches every parameter" do
    model = Llama.load(dir)
    ids = Tensor.from(expected.prompt_ids.first(9), [1, 9])
    loss = model.call(ids.slice(1, 0, 8)).cross_entropy(ids.slice(1, 1, 9).reshape([8]).contiguous)
    grads = Tensomite.grad(loss, model.params.map(&.value))
    grads.each { |grad| grad.to_a(Float32).any?(&.!=(0)).should be_true }
  end

  it "raises when positions exceed the context" do
    model = Llama.load(dir)
    expect_raises(ShapeError) { model.forward(Array.new(65, 1), model.new_cache) }
  end

  it "rejects configurations it would run wrongly" do
    config = JSON.parse(File.read("#{dir}/config.json")).as_h
    config["rope_scaling"] = JSON.parse(%({"rope_type": "llama3", "factor": 8.0}))
    expect_raises(Error, /rope_scaling/) { Llama::Config.from_json(config.to_json) }
  end
end

describe "M9 exit: SmolLM2-135M-Instruct" do
  dir = ENV["TENSOMITE_SMOLLM2_DIR"]? || ""
  if File.exists?("#{dir}/model.safetensors")
    expected = LlamaExpected.read("#{__DIR__}/fixtures/smollm2/logits.json")
    bpe = BPE.from_json("#{dir}/tokenizer.json")

    it "tokenises the ChatML prompt like the reference" do
      bpe.encode(expected.prompt).should eq expected.prompt_ids
    end

    (DEVICES + GPUS).each do |device|
      describe "on #{device.class}" do
        model = Llama.load(dir, device)

        it "matches the reference logits within 2e-3" do
          expected.error(model.forward(expected.prompt_ids, model.new_cache(64))).should be < 2e-3
        end

        it "answers greedily like the reference" do
          model.generate(expected.prompt_ids, expected.greedy.size).should eq expected.greedy
        end

        if device.shape_invariant?
          it "gives bit-identical logits in prefill and in one-by-one decode" do
            ids = expected.prompt_ids
            prefill = model.forward(ids, model.new_cache(ids.size)).to_a(Float32)
            cache = model.new_cache(ids.size)
            decoded = [] of Float32
            ids.each { |id| decoded = model.forward([id], cache).to_a(Float32) }
            decoded.should eq prefill[-model.config.vocab..]
          end
        end
      end
    end
  else
    pending "needs TENSOMITE_SMOLLM2_DIR (see scripts/fetch_smollm2.sh)"
  end
end
