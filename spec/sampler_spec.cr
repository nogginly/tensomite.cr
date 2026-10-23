require "./spec_helper"

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

  it "rejects a negative temperature" do
    expect_raises(ArgumentError) { Sampler.new(temperature: -1.0) }
  end
end
