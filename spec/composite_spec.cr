require "./spec_helper"

# Expected values for transcendental functions were computed independently in
# Python's `math` module, in float64.
DEVICES.each do |device|
  describe "Composite ops on #{device.class}" do
    describe "softmax" do
      it "normalises along the last axis" do
        x = tensor(device, [2, 3], f32(1, 2, 3, 1, 2, 3))
        expect_close x.softmax.to_a(Float32), f32(0.0900306, 0.2447285, 0.6652410, 0.0900306, 0.2447285, 0.6652410)
      end

      it "normalises along another axis" do
        # columns of [[1, 1], [2, 2], [3, 3]]
        x = tensor(device, [3, 2], f32(1, 1, 2, 2, 3, 3))
        expect_close x.softmax(0).to_a(Float32), f32(0.0900306, 0.0900306, 0.2447285, 0.2447285, 0.6652410, 0.6652410)
      end

      it "does not overflow on large inputs" do
        x = tensor(device, [2], f32(1000, 1001))
        expect_close x.softmax.to_a(Float32), f32(0.2689414, 0.7310586)
      end
    end

    describe "layernorm" do
      x = tensor(device, [1, 3], f32(1, 2, 3))

      it "centres and scales each row" do
        # mean 2, variance 2/3, so (x - 2) / sqrt(2/3)
        ones = tensor(device, [3], f32(1, 1, 1))
        zeros = tensor(device, [3], f32(0, 0, 0))
        expect_close x.layernorm(ones, zeros, eps: 0).to_a(Float32), f32(-1.2247449, 0, 1.2247449)
      end

      it "applies weight and bias" do
        w = tensor(device, [3], f32(2, 2, 2))
        b = tensor(device, [3], f32(1, 1, 1))
        expect_close x.layernorm(w, b, eps: 0).to_a(Float32), f32(-1.4494897, 1, 3.4494897)
      end
    end

    it "computes GELU with the tanh approximation" do
      x = tensor(device, [4], f32(-1, 0, 1, 2))
      expect_close x.gelu.to_a(Float32), f32(-0.158808, 0, 0.841192, 1.9545977)
    end

    describe "causal_mask" do
      inf = Float32::INFINITY

      it "hides future positions" do
        Tensor.causal_mask(3, 3, device).to_a(Float32).should eq [0, -inf, -inf, 0, 0, -inf, 0, 0, 0]
      end

      it "lets queries see cached earlier positions" do
        # one cached position: row i sees columns 0..i + 1
        Tensor.causal_mask(2, 3, device).to_a(Float32).should eq [0, 0, -inf, 0, 0, 0]
        Tensor.causal_mask(1, 3, device).to_a(Float32).should eq f32(0, 0, 0)
      end

      it "turns masked scores into zero attention weight" do
        scores = tensor(device, [3, 3], f32(1, 2, 3, 1, 2, 3, 1, 2, 3))
        weights = (scores + Tensor.causal_mask(3, 3, device)).softmax
        expect_close weights.to_a(Float32), f32(1, 0, 0, 0.2689414, 0.7310586, 0, 0.0900306, 0.2447285, 0.6652410)
      end

      it "rejects more rows than columns" do
        expect_raises(ShapeError) { Tensor.causal_mask(3, 2, device) }
      end
    end
  end
end
