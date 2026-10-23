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
        (a.transpose * a + 1).exp.realize
        metal.gpu_ops.should eq gpu + 3
        metal.host_ops.should eq host
        # a·a = [[7, 10], [15, 22]], which sums to 54; only the matmul runs on the host.
        a.matmul(a).sum.to_a(Float32).should eq f32(54)
        metal.host_ops.should eq host + 1
        metal.gpu_ops.should eq gpu + 4
      end

      it "mixes GPU and host ops in one graph" do
        a = Tensor.from(f32(1, 2, 3, 4), [2, 2], metal)
        # a·a = [[7, 10], [15, 22]]; + a = [[8, 12], [18, 26]]; summed by rows = [20, 44]
        (a.matmul(a) + a).sum(1).to_a(Float32).should eq f32(20, 44)
      end

      it "shares memory with the host" do
        t = Tensor.from(f32(1, 2), [2], metal)
        metal.map(t.node.buffer!).should_not be_nil
      end
    end
  end
{% end %}
