require "./spec_helper"

# The sampler as first written, sorting every candidate: the oracle for its
# choices on logits without ties.
private class SortingSampler
  def initialize(@temperature : Float64, @top_k : Int32, seed : Int32)
    @random = Random::PCG32.new(seed.to_u64)
  end

  def pick(logits : Array(Float32)) : Int32
    candidates = (0...logits.size).to_a
    candidates = candidates.sort_by { |i| -logits[i] }.first(@top_k) if 0 < @top_k < logits.size
    top = candidates.max_of { |i| logits[i] }.to_f64
    weights = candidates.map { |i| Math.exp((logits[i] - top) / @temperature) }
    draw = @random.rand * weights.sum
    candidates.each_with_index do |token, rank|
      draw -= weights[rank]
      return token if draw < 0
    end
    candidates.last
  end
end

describe Sampler do
  logits = f32(1, 5, 3, 5, -2)

  it "picks the first highest logit when greedy" do
    Sampler.greedy.pick(logits).should eq 1
  end

  it "repeats a run for the same seed" do
    picks = ->(seed : Int32) { sampler = Sampler.new(seed: seed); Array.new(50) { sampler.pick(logits) } }
    picks.call(7).should eq picks.call(7)
    picks.call(7).should_not eq picks.call(8)
  end

  it "only picks among the top k" do
    sampler = Sampler.new(top_k: 2, seed: 1)
    Array.new(200) { sampler.pick(logits) }.uniq.sort!.should eq [1, 3]
  end

  it "concentrates on the highest logits at low temperature" do
    sampler = Sampler.new(temperature: 0.05, seed: 3)
    Array.new(100) { sampler.pick(logits) }.uniq.sort!.should eq [1, 3]
  end

  it "picks the tokens the sorting sampler picks, for the same seed, on logits without ties" do
    rng = Random.new(5)
    rows = Array.new(20) { (0...3000).to_a.shuffle(rng).map { |v| (v * 0.004 - 6).to_f32 } }
    [{0.8, 40}, {1.0, 0}, {0.5, 1}, {1.3, 2999}, {0.9, 3000}].each do |(temperature, top_k)|
      fast, sorting = Sampler.new(temperature: temperature, top_k: top_k, seed: 11), SortingSampler.new(temperature, top_k, 11)
      rows.each { |row| 5.times { fast.pick(row).should eq sorting.pick(row) } }
    end
  end

  it "breaks ties in the top k by the lower index" do
    sampler = Sampler.new(top_k: 2, seed: 2)
    Array.new(200) { sampler.pick(f32(5, 1, 5, 5, 0)) }.uniq.sort!.should eq [0, 2]
  end

  it "rejects a negative temperature" do
    expect_raises(ArgumentError) { Sampler.new(temperature: -1.0) }
  end
end
