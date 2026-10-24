require "./spec_helper"

describe BufferPool do
  it "returns memory of exactly the size given back" do
    pool = BufferPool(Bytes).new(1 << 20)
    memory = Bytes.new(16)
    pool.give(memory, 16).should be_true
    pool.take(8).should be_nil
    pool.take(16).should eq memory
    pool.take(16).should be_nil
    {pool.hits, pool.misses, pool.held}.should eq({1, 2, 0})
  end

  it "declines memory beyond its limit, and empty memory" do
    pool = BufferPool(Bytes).new(24)
    pool.give(Bytes.new(16), 16).should be_true
    pool.give(Bytes.new(16), 16).should be_false
    pool.give(Bytes.new(0), 0).should be_false
    pool.held.should eq 16
  end
end

describe HostDevice do
  it "reuses the memory of collected buffers" do
    device = CPU.new
    20.times { (Tensor.from(f32(1, 2, 3, 4), [4], device) * 2).realize }
    5.times { GC.collect }
    hits = device.pool.hits
    20.times { (Tensor.from(f32(1, 2, 3, 4), [4], device) * 2).to_a(Float32).should eq f32(2, 4, 6, 8) }
    device.pool.hits.should be > hits
  end
end
