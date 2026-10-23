require "./spec_helper"
require "json"

# Reference outputs from scripts/make_fixtures.py: transformers' Python GPT-2
# tokeniser and PyTorch's GPT-2, run on the same files.
private record Expected, corpus : Array(String), corpus_ids : Array(Array(Int32)),
  prompt : String, prompt_ids : Array(Int32), logit_rows : Array(Int32),
  logits : Array(Array(Float32)), greedy : Array(Int32) do
  def self.read(path : String) : Expected
    json = JSON.parse(File.read(path))
    ints = ->(any : JSON::Any) { any.as_a.map(&.as_i) }
    new(
      json["corpus"].as_a.map(&.as_s), json["corpus_ids"].as_a.map { |row| ints.call(row) },
      json["prompt"].as_s, ints.call(json["prompt_ids"]), ints.call(json["logit_rows"]),
      json["logits"].as_a.map { |row| row.as_a.map(&.as_f.to_f32) }, ints.call(json["greedy"]))
  end
end

# Returns the largest absolute difference between the expected logit rows and *logits*.
private def logit_error(expected : Expected, logits : Tensor) : Float32
  width = logits.shape[1]
  values = logits.to_a(Float32)
  expected.logit_rows.zip(expected.logits).max_of do |row, want|
    want.each_with_index.max_of { |value, col| (values[row * width + col] - value).abs }
  end
end

# Runs *ids* one position at a time and returns the logits of every position.
private def decode_one_by_one(model : GPT2, ids : Array(Int32)) : Array(Float32)
  cache = model.new_cache
  ids.flat_map { |id| model.forward([id], cache).to_a(Float32) }
end

describe "GPT-2 on the tiny reference model" do
  dir = "#{__DIR__}/fixtures/tiny_gpt2"
  expected = Expected.read("#{dir}/expected.json")
  bpe = BPE.from_files("#{dir}/vocab.json", "#{dir}/merges.txt")

  it "tokenises the Unicode corpus like the reference" do
    expected.corpus.map { |text| bpe.encode(text) }.should eq expected.corpus_ids
    bpe.encode(expected.prompt).should eq expected.prompt_ids
  end

  it "round-trips the corpus through encode and decode" do
    expected.corpus.each { |text| bpe.decode(bpe.encode(text)).should eq text }
  end

  (DEVICES + GPUS).each do |device|
    describe "on #{device.class}" do
      model = GPT2.load("#{dir}/model.safetensors", heads: 4, device: device)

      it "reads its configuration from the weights" do
        model.config.should eq GPT2::Config.new(layers: 2, heads: 4, width: 32, context: 64, vocab: 400)
      end

      it "matches the reference logits within 1e-4" do
        logits = model.forward(expected.prompt_ids, model.new_cache)
        logits.shape.should eq [expected.prompt_ids.size, model.config.vocab]
        logit_error(expected, logits).should be <= 1e-4
      end

      it "decodes greedily like the reference" do
        model.generate(expected.prompt_ids, expected.greedy.size).should eq expected.greedy
      end

      if device.shape_invariant?
        it "gives bit-identical logits in prefill and in one-by-one decode" do
          prefill = model.forward(expected.prompt_ids, model.new_cache).to_a(Float32)
          decode_one_by_one(model, expected.prompt_ids).should eq prefill
        end
      end
    end
  end

  it "raises when positions exceed the context" do
    model = GPT2.load("#{dir}/model.safetensors", heads: 4)
    expect_raises(ShapeError, /exceed the context/) { model.forward(Array.new(65, 0), model.new_cache) }
    expect_raises(ShapeError) { model.forward([] of Int32, model.new_cache) }
  end
end

# The M4 exit criterion, on GPT-2 small. It needs the weights and tokeniser files
# in TENSOMITE_GPT2_DIR (see scripts/fetch_gpt2.sh) and fixtures generated from
# them in spec/fixtures/gpt2.
describe "M4 exit: GPT-2 small" do
  dir = ENV["TENSOMITE_GPT2_DIR"]? || ""
  fixture = "#{__DIR__}/fixtures/gpt2/expected.json"
  ready = File.exists?("#{dir}/model.safetensors") && File.exists?(fixture)

  if ready
    expected = Expected.read(fixture)
    bpe = BPE.from_files("#{dir}/vocab.json", "#{dir}/merges.txt")
    model = GPT2.load("#{dir}/model.safetensors")

    it "tokenises the Unicode corpus like the reference" do
      expected.corpus.map { |text| bpe.encode(text) }.should eq expected.corpus_ids
      bpe.encode(expected.prompt).should eq expected.prompt_ids
    end

    it "matches the reference logits within 2e-3" do
      logit_error(expected, model.forward(expected.prompt_ids, model.new_cache)).should be <= 2e-3
    end

    it "decodes 64 tokens greedily like the reference" do
      model.generate(expected.prompt_ids, 64).should eq expected.greedy
    end

    it "gives bit-identical logits in prefill and in one-by-one decode" do
      prefill = model.forward(expected.prompt_ids, model.new_cache).to_a(Float32)
      decode_one_by_one(model, expected.prompt_ids).should eq prefill
    end
  else
    pending "needs TENSOMITE_GPT2_DIR and spec/fixtures/gpt2/expected.json"
  end
end

# The M6 exit criterion on GPT-2 small: the same checks as M4, on each GPU.
describe "M6 exit: GPT-2 small on GPUs" do
  dir = ENV["TENSOMITE_GPT2_DIR"]? || ""
  fixture = "#{__DIR__}/fixtures/gpt2/expected.json"
  ready = File.exists?("#{dir}/model.safetensors") && File.exists?(fixture)

  if !ready
    pending "needs TENSOMITE_GPT2_DIR and spec/fixtures/gpt2/expected.json"
  elsif GPUS.empty?
    pending "no GPU on this machine"
  else
    expected = Expected.read(fixture)
    GPUS.each do |gpu|
      describe gpu.class do
        model = GPT2.load("#{dir}/model.safetensors", device: gpu)

        it "matches the reference logits within 2e-3" do
          logit_error(expected, model.forward(expected.prompt_ids, model.new_cache)).should be <= 2e-3
        end

        it "decodes 64 tokens greedily like the reference" do
          model.generate(expected.prompt_ids, 64).should eq expected.greedy
        end

        it "gives bit-identical logits in prefill and in one-by-one decode" do
          prefill = model.forward(expected.prompt_ids, model.new_cache).to_a(Float32)
          decode_one_by_one(model, expected.prompt_ids).should eq prefill
        end
      end
    end
  end
end
