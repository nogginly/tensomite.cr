require "./spec_helper"

private def tiny_gpt2(device : Device) : GPT2
  GPT2.new(GPT2::Config.new(layers: 2, heads: 4, width: 32, context: 64, vocab: 50), Builder.new(rng: Random.new(1), device: device))
end

private def tiny_llama(device : Device, merged : Bool = false) : Llama
  config = Llama::Config.new(layers: 2, heads: 4, kv_heads: 2, width: 32, hidden: 48, head_width: 8, context: 64, vocab: 50)
  model = Llama.new(config, Builder.new(rng: Random.new(2), device: device))
  merged ? model.merge_projections : model
end

# Each step's appends for three rows of different lengths: prompts, then
# ragged appends with a row idle, then one token each.
STEPS = [
  [[1, 2, 3, 4, 5], [6, 7], [8, 9, 10, 11, 12, 13, 14, 15, 16]],
  [[17, 18, 19], [] of Int32, [20]],
  [[21], [22], [23]],
  [[] of Int32, [24, 25, 26, 27], [28, 29]],
]

# Returns each step's logits for each row decoded alone, row by row, `nil`
# where the row was idle.
private def alone(model, last : Bool) : Array(Array(Array(Float32)?))
  caches = Array.new(3) { model.new_cache }
  STEPS.map do |step|
    step.map_with_index do |ids, row|
      next if ids.empty?
      (last ? model.forward_last(ids, caches[row]) : model.forward(ids, caches[row])).to_a(Float32)
    end
  end
end

# Checks *got* against *expected*: bit for bit on a shape-invariant device,
# and within rounding on one, such as BLAS, whose products depend on their
# number of rows.
private def same(got : Array(Float32), expected : Array(Float32), device : Device)
  if device.shape_invariant?
    got.should eq expected
  else
    got.size.should eq expected.size
    got.zip(expected) { |a, e| a.should be_close(e, 1e-4) }
  end
end

private def check_rows(model, vocab : Int32)
  expected = alone(model, last: true)
  cache = model.new_batch_cache(3)
  STEPS.each_with_index do |step, i|
    got = model.forward_last(step, cache).to_a(Float32)
    active = (0...3).select { |row| !step[row].empty? }
    got.size.should eq active.size * vocab
    active.each_with_index do |row, line|
      same(got[line * vocab, vocab], expected[i][row].as(Array(Float32)), model.device)
    end
  end
  3.times { |row| cache.length(row).should eq STEPS.sum(&.[row].size) }
end

describe "Batched decoding" do
  (DEVICES + GPUS).each do |device|
    it "gives each GPT-2 row its logits alone on #{device.class}, appended raggedly" do
      check_rows(tiny_gpt2(device), 50)
    end

    it "gives each Llama row its logits alone on #{device.class}, appended raggedly" do
      check_rows(tiny_llama(device), 50)
      check_rows(tiny_llama(device, merged: true), 50)
    end

    it "gives every new position's logits, the rows' one after another, on #{device.class}" do
      model = tiny_llama(device)
      expected = alone(model, last: false)
      cache = model.new_batch_cache(3)
      STEPS.each_with_index do |step, i|
        same(model.forward(step, cache).to_a(Float32), expected[i].compact.flatten, device)
      end
    end
  end

  it "starts a reset row afresh, keeping the others" do
    model = tiny_gpt2(CPU.default)
    cache = model.new_batch_cache(2, capacity: 32)
    model.forward_last([[1, 2, 3], [4, 5]], cache)
    cache.reset(0)
    cache.length(0).should eq 0
    got = model.forward_last([[7, 8], [6]], cache).to_a(Float32)
    fresh = model.forward_last([7, 8], model.new_cache(32)).to_a(Float32)
    other = model.new_cache(32)
    model.forward([4, 5], other)
    got[0, 50].should eq fresh
    got[50, 50].should eq model.forward_last([6], other).to_a(Float32)
  end

  it "raises for appends that do not match the rows, or hold no tokens" do
    model = tiny_gpt2(CPU.default)
    cache = model.new_batch_cache(2)
    expect_raises(ShapeError, /3 appends for 2 rows/) { model.forward_last([[1], [2], [3]], cache) }
    expect_raises(ShapeError, /no tokens/) { model.forward_last([[] of Int32, [] of Int32], cache) }
  end
end
