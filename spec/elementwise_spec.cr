require "./spec_helper"

DEVICES.each do |device|
  describe "Elementwise ops on #{device.class}" do
    # a = [[1, 2, 3], [4, 5, 6]], b = [[6, 5, 4], [3, 2, 1]]
    a = tensor(device, [2, 3], f32(1, 2, 3, 4, 5, 6))
    b = tensor(device, [2, 3], f32(6, 5, 4, 3, 2, 1))

    it "subtracts, divides and takes the maximum" do
      (a - b).to_a(Float32).should eq f32(-5, -3, -1, 1, 3, 5)
      (a / b).to_a(Float32).should eq [1f32 / 6, 2f32 / 5, 3f32 / 4, 4f32 / 3, 5f32 / 2, 6f32]
      a.maximum(b).to_a(Float32).should eq f32(6, 5, 4, 4, 5, 6)
    end

    it "applies scalars and negation" do
      (a + 1).to_a(Float32).should eq f32(2, 3, 4, 5, 6, 7)
      (a - 1).to_a(Float32).should eq f32(0, 1, 2, 3, 4, 5)
      (a * 0.5).to_a(Float32).should eq f32(0.5, 1, 1.5, 2, 2.5, 3)
      (a / 2).to_a(Float32).should eq f32(0.5, 1, 1.5, 2, 2.5, 3)
      (-a).to_a(Float32).should eq f32(-1, -2, -3, -4, -5, -6)
    end

    it "divides a transposed view by a broadcast row" do
      # a.T = [[1, 4], [2, 5], [3, 6]], divided by [1, 2]
      (a.transpose / tensor(device, [2], f32(1, 2))).to_a(Float32).should eq f32(1, 2, 2, 2.5, 3, 3)
    end

    it "applies unary maths" do
      expect_close tensor(device, [2], f32(0, 1)).exp.to_a(Float32), f32(1, 2.718281828)
      expect_close tensor(device, [2], f32(1, 2)).log.to_a(Float32), f32(0, 0.693147181)
      expect_close tensor(device, [2], f32(0, 1)).tanh.to_a(Float32), f32(0, 0.761594156)
      expect_close tensor(device, [2], f32(4, 2)).sqrt.to_a(Float32), f32(2, 1.414213562)
    end

    it "keeps I32 arithmetic exact" do
      i = tensor(device, [3], [7, -2, 5])
      (i - 1).to_a(Int32).should eq [6, -3, 4]
      i.maximum(tensor(device, [1], [0])).to_a(Int32).should eq [7, 0, 5]
    end

    it "rejects float-only ops on I32 when the op is called" do
      i = tensor(device, [2], [1, 2])
      expect_raises(DTypeError) { i.exp }
      expect_raises(DTypeError) { i / i }
      expect_raises(DTypeError) { i * 0.5 }
    end
  end
end
