require "./spec_helper"

describe "Tensor#dropout" do
  DEVICES.each do |device|
    describe "on #{device.class}" do
      x = Tensor.from(random_values(20_000, 1).map { |v| v + 2 }, [100, 200], device)

      it "zeroes about its rate of elements and scales the rest by 1 / (1 - rate)" do
        values, kept = x.dropout(0.25, seed: 7_u64).to_a(Float32), x.to_a(Float32)
        scale = (1 / 0.75).to_f32
        values.zip(kept).each { |(result, input)| (result == 0 || result == input * scale).should be_true }
        (values.count(0f32) / values.size).should be_close(0.25, 0.02)
      end

      it "gives the same mask for the same seed, through any view, and another for another seed" do
        x.dropout(0.5, seed: 3_u64).to_a(Float32).should eq x.dropout(0.5, seed: 3_u64).to_a(Float32)
        x.dropout(0.5, seed: 3_u64).to_a(Float32).should_not eq x.dropout(0.5, seed: 4_u64).to_a(Float32)
        xt = x.transpose
        xt.dropout(0.5, seed: 3_u64).to_a(Float32).should eq xt.contiguous.dropout(0.5, seed: 3_u64).to_a(Float32)
      end

      it "passes its gradient through the same mask and scale" do
        w = Tensor.from(random_values(20_000, 2), [100, 200], device)
        dx = Tensomite.grad((x.dropout(0.3, seed: 9_u64) * w).sum, [x]).first
        dx.to_a(Float32).should eq w.dropout(0.3, seed: 9_u64).to_a(Float32)
      end
    end
  end

  it "returns the tensor itself at rate 0, and rejects rates outside [0, 1)" do
    x = Tensor.from(f32(1, 2, 3), [3])
    x.dropout(0.0, seed: 1_u64).node.should be x.node
    expect_raises(ArgumentError) { x.dropout(1.0, seed: 1_u64) }
    expect_raises(ArgumentError) { x.dropout(-0.1, seed: 1_u64) }
  end
end

describe Dropout do
  it "gives each use its own mask, the same again for the same seed" do
    x = Tensor.from(random_values(1000, 3), [1000])
    first = Dropout.new(0.5, seed: 11_u64)
    a, b = first.call(x).to_a(Float32), first.call(x).to_a(Float32)
    a.should_not eq b
    again = Dropout.new(0.5, seed: 11_u64)
    again.call(x).to_a(Float32).should eq a
    again.call(x).to_a(Float32).should eq b
  end

  it "changes GPT-2's training logits, and leaves them alone without it" do
    model = GPT2.load("#{__DIR__}/fixtures/tiny_gpt2/model.safetensors", heads: 4)
    ids = Tensor.from((1..16).to_a, [2, 8])
    plain = model.call(ids).to_a(Float32)
    model.call(ids, dropout: Dropout.new(0.1, seed: 5_u64)).to_a(Float32).should_not eq plain
    model.call(ids, dropout: Dropout.new(0.1, seed: 5_u64)).to_a(Float32).should eq model.call(ids, dropout: Dropout.new(0.1, seed: 5_u64)).to_a(Float32)
    model.call(ids).to_a(Float32).should eq plain
  end
end
