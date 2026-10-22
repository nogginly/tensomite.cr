require "./spec_helper"

DEVICES.each do |device|
  describe "Reductions on #{device.class}" do
    # a = [[1, 2, 3], [4, 5, 6]]
    a = tensor(device, [2, 3], f32(1, 2, 3, 4, 5, 6))

    it "takes the maximum overall and over an axis" do
      a.max.to_a(Float32).should eq f32(6)
      a.max(0).to_a(Float32).should eq f32(4, 5, 6)
      a.max(1).to_a(Float32).should eq f32(3, 6)
      # rows of a.T: [1, 4], [2, 5], [3, 6]
      a.transpose.max(1).to_a(Float32).should eq f32(4, 5, 6)
    end

    it "starts the maximum from the first element, not from zero" do
      tensor(device, [3], f32(-3, -1, -2)).max.to_a(Float32).should eq f32(-1)
      tensor(device, [3], [3, 7, 5]).max.to_a(Int32).should eq [7]
    end

    it "takes the mean" do
      a.mean.to_a(Float32).should eq f32(3.5)
      a.mean(0).to_a(Float32).should eq f32(2.5, 3.5, 4.5)
      a.mean(-1).to_a(Float32).should eq f32(2, 5)
    end

    it "keeps reduced dimensions with size 1" do
      kept = a.sum(1, keepdims: true)
      kept.shape.should eq [2, 1]
      kept.to_a(Float32).should eq f32(6, 15)
      a.max(keepdims: true).shape.should eq [1, 1]
    end

    it "raises when the op is called on invalid input" do
      expect_raises(ShapeError) { tensor(device, [2, 0], [] of Float32).max(1) }
      expect_raises(DTypeError) { tensor(device, [2], [1, 2]).mean }
    end
  end
end
