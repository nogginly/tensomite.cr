require "./spec_helper"

describe SimDiscrete do
  sim = SimDiscrete.default
  cpu = CPU.default

  it "is a separate device from the CPU" do
    sim.should_not be cpu
  end

  it "never exposes its memory to the host" do
    t = tensor(sim, [2], f32(1, 2))
    sim.map(t.node.buffer!).should be_nil
  end

  it "moves data only through counted transfers" do
    uploads, downloads = sim.uploads, sim.downloads
    t = tensor(sim, [3], f32(1, 2, 3))
    (t + t).to_a(Float32).should eq f32(2, 4, 6)
    (sim.uploads - uploads).should eq 1
    (sim.downloads - downloads).should eq 1
  end

  it "rejects buffers belonging to another device" do
    t = tensor(cpu, [1], f32(1))
    expect_raises(DeviceError) { sim.download(t.node.buffer!, Bytes.new(4)) }
  end
end

describe "Tensor#to" do
  cpu = CPU.default
  sim = SimDiscrete.default

  it "copies a tensor between devices in logical order" do
    # a.T = [[1, 4], [2, 5], [3, 6]]
    a = tensor(cpu, [2, 3], f32(1, 2, 3, 4, 5, 6))
    moved = a.transpose.to(sim)
    moved.device.should be sim
    moved.shape.should eq [3, 2]
    moved.to(cpu).to_a(Float32).should eq f32(1, 4, 2, 5, 3, 6)
  end

  it "returns the same tensor when already on the device" do
    a = tensor(cpu, [1], f32(1))
    a.to(cpu).node.should be a.node
  end

  it "refuses ops that mix devices" do
    expect_raises(DeviceError) { tensor(cpu, [1], f32(1)) + tensor(sim, [1], f32(1)) }
  end
end
