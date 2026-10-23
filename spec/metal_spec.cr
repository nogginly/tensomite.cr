require "./spec_helper"

{% if flag?(:darwin) && flag?(:aarch64) %}
  describe Metal do
    pending "no Metal GPU on this machine" unless Metal.available?

    if Metal.available?
      metal = Metal.default

      it "names the GPU" do
        metal.name.should_not be_empty
      end

      it "runs ported ops on the GPU and the rest on the host" do
        gpu, host = metal.gpu_ops, metal.host_ops
        a = Tensor.from(f32(1, 2, 3, 4), [2, 2], metal)
        # a·a = [[7, 10], [15, 22]], which sums to 54.
        a.matmul(a).sum.to_a(Float32).should eq f32(54)
        metal.gpu_ops.should eq gpu + 2
        metal.host_ops.should eq host
        # I32 has no Metal kernels, so it runs on the host.
        (Tensor.from([1, 2], [2], metal) + Tensor.from([3, 4], [2], metal)).to_a(Int32).should eq [4, 6]
        metal.host_ops.should eq host + 1
      end

      it "mixes GPU and host ops in one graph" do
        a = Tensor.from(f32(1, 2, 3, 4), [2, 2], metal)
        ids = Tensor.from([1, 0], [2], metal) * Tensor.from([1, 1], [2], metal)
        # ids (I32, on the host) = [1, 0]; rows of a in that order = [[3, 4], [1, 2]];
        # · a = [[15, 22], [7, 10]]; summed by rows = [37, 17]
        a.gather(ids).matmul(a).sum(1).to_a(Float32).should eq f32(37, 17)
      end

      it "shares memory with the host" do
        t = Tensor.from(f32(1, 2), [2], metal)
        metal.map(t.node.buffer!).should_not be_nil
      end
    end
  end
{% end %}
