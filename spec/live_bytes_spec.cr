require "./spec_helper"

describe "Device#live_bytes and #peak_live_bytes" do
  (DEVICES + GPUS).each do |device|
    describe device.class do
      it "counts a realised tensor's bytes, and keeps the peak when it is released" do
        before = device.live_bytes
        device.reset_peak
        t = (Tensor.full([1024, 256], 1, device: device) * 2).realize
        bytes = t.node.bytesize
        device.live_bytes.should be >= before + bytes
        device.peak_live_bytes.should be >= before + bytes
        peak = device.peak_live_bytes
        t.node.buffer!.release
        device.live_bytes.should be <= peak - bytes
        device.peak_live_bytes.should eq peak
      end

      it "gives a step's own peak after reset_peak" do
        big = (Tensor.full([2048, 512], 1, device: device) * 3).realize
        big.node.buffer!.release
        device.reset_peak
        device.peak_live_bytes.should eq device.live_bytes
        small = (Tensor.full([16, 16], 1, device: device) * 3).realize
        device.peak_live_bytes.should be < device.live_bytes + big.node.bytesize
        small.node.buffer!.release
      end

      it "counts an intermediate freed inside one realize" do
        x = Tensor.full([512, 512], 1, device: device)
        root = ((x + x) * 2 + x).sum
        device.reset_peak
        start = device.live_bytes
        Tensor.realize([root], release: true)
        device.peak_live_bytes.should be >= start + x.node.bytesize
        device.live_bytes.should be < start + x.node.bytesize
      end
    end
  end

  it "leaves out a tensor mapped from a file" do
    path = File.tempname("tensomite", ".safetensors")
    begin
      SafeTensors.write(path, {"w" => Tensor.full([256, 256], 1)})
      before = CPU.default.live_bytes
      loaded = SafeTensors.open(path)["w"]
      CPU.default.live_bytes.should eq before
      loaded.sum.to_a(Float32).should eq [65536f32]
    ensure
      File.delete?(path)
    end
  end
end
