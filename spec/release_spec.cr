require "./spec_helper"

# A chain of elementwise ops whose intermediates all have the size of *x*.
private def chain(x : Tensor) : Tensor
  y = x
  6.times { |i| y = (y + x) * (i + 1).to_f32 }
  y
end

describe "Releasing intermediates while realising" do
  (DEVICES + GPUS).each do |device|
    describe device.class do
      it "gives the bits of an ordinary realize, and keeps the roots readable" do
        x = Tensor.from(f32(1, 2, 3, 4), [4], device)
        plain = chain(x).to_a(Float32)
        root = chain(x)
        Tensor.realize([root], release: true)
        root.to_a(Float32).should eq plain
        x.to_a(Float32).should eq f32(1, 2, 3, 4)
      end

      it "releases intermediates, which then raise when read" do
        x = Tensor.from(f32(1, 2, 3, 4), [4], device)
        middle = (x + x) * 2.0_f32
        root = middle.exp + middle
        Tensor.realize([root], release: true)
        middle.node.released?.should be_true
        error = expect_raises(Error) { middle.to_a(Float32) }
        error.message.to_s.should contain "released"
        root.to_a(Float32).should eq ((x + x) * 2.0_f32).exp.to_a(Float32).zip(((x + x) * 2.0_f32).to_a(Float32)).map { |a, b| a + b }
      end

      it "keeps an intermediate that is also a root" do
        x = Tensor.from(f32(1, 2), [2], device)
        middle = x * 3.0_f32
        root = middle + 1.0_f32
        Tensor.realize([root, middle], release: true)
        middle.to_a(Float32).should eq f32(3, 6)
        root.to_a(Float32).should eq f32(4, 7)
      end

      it "never releases a buffer that a write fills" do
        cache = Tensor.zeros([2, 2], device: device).realize
        row = Tensor.from(f32(1, 2), [1, 2], device) * 2.0_f32
        written = cache.write(row, 0, 1)
        result = written * 1.0_f32
        Tensor.realize([result], release: true)
        written.node.released?.should be_false
        cache.to_a(Float32).should eq f32(0, 0, 2, 4)
        result.to_a(Float32).should eq f32(0, 0, 2, 4)
      end
    end
  end

  it "reuses released memory later in the same realize" do
    device = CPU.new
    x = Tensor.from(Array.new(1024, &.to_f32), [1024], device)
    hits = device.pool.hits
    Tensor.realize([chain(x)], release: true)
    (device.pool.hits - hits).should be > 0
  end
end

describe "Optimizer#step(loss)" do
  it "releases the step's intermediates but returns the loss" do
    param = Parameter.new(Tensor.from(f32(1, 2, 3), [3]))
    optimizer = AdamW.new([param], lr: 0.1)
    hidden = param.value * 2.0_f32
    loss = (hidden * hidden).sum
    optimizer.step(loss).should eq 56.0_f32
    hidden.node.released?.should be_true
    loss.to_a(Float32).should eq f32(56)
  end
end
