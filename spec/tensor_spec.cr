require "./spec_helper"

DEVICES.each do |device|
  describe "Tensor on #{device.class}" do
    # a = [[1, 2, 3],
    #      [4, 5, 6]]
    a = tensor(device, [2, 3], f32(1, 2, 3, 4, 5, 6))

    describe "M1 exit: (a + b).sum over views" do
      # a.T = [[1, 4], [2, 5], [3, 6]]
      it "sums a transposed view plus a contiguous tensor" do
        b = tensor(device, [3, 2], f32(10, 20, 30, 40, 50, 60))
        # a.T + b = [[11, 24], [32, 45], [53, 66]]
        (a.transpose + b).sum.to_a(Float32).should eq f32(231)
        (a.transpose + b).sum(0).to_a(Float32).should eq f32(96, 135)
        (a.transpose + b).sum(1).to_a(Float32).should eq f32(35, 77, 119)
      end

      it "sums a transposed view plus a broadcast row" do
        c = tensor(device, [2], f32(100, 200))
        # a.T + c = [[101, 204], [102, 205], [103, 206]]
        (a.transpose + c).sum.to_a(Float32).should eq f32(921)
        (a.transpose + c).sum(0).to_a(Float32).should eq f32(306, 615)
      end

      it "multiplies a transposed view by a broadcast column" do
        d = tensor(device, [3, 1], f32(1, 2, 3))
        # a.T * d = [[1, 4], [4, 10], [9, 18]]
        (a.transpose * d).sum.to_a(Float32).should eq f32(46)
      end

      it "adds two transposed views of the same node" do
        t = a.transpose
        # rows of a.T doubled: [[2, 8], [4, 10], [6, 12]]
        (t + t).sum(-1).to_a(Float32).should eq f32(10, 14, 18)
      end
    end

    it "round-trips data and shape" do
      a.shape.should eq [2, 3]
      a.dtype.should eq DType::F32
      a.device.should be device
      a.to_a(Float32).should eq f32(1, 2, 3, 4, 5, 6)
    end

    it "reads views in logical order" do
      a.transpose.to_a(Float32).should eq f32(1, 4, 2, 5, 3, 6)
      tensor(device, [1], f32(7)).expand([2, 2]).to_a(Float32).should eq f32(7, 7, 7, 7)
    end

    it "supports I32" do
      t = tensor(device, [3], [1, 2, 3])
      (t * t).sum.to_a(Int32).should eq [14]
    end

    it "does no work until realised" do
      s = (a + a).sum
      s.node.realized?.should be_false
      s.realize.node.realized?.should be_true
    end

    it "drops a node's sources once realised" do
      s = (a + a).sum
      s.node.srcs.size.should eq 1
      s.realize.node.srcs.should be_empty
    end

    it "computes a shared source once" do
      x = a + a
      y = x * x
      y.to_a(Float32).should eq f32(4, 16, 36, 64, 100, 144)
      x.node.realized?.should be_true
    end

    it "reshapes contiguous tensors without copying and others with a copy" do
      a.reshape([3, 2]).node.should be a.node
      r = a.transpose.reshape([6])
      r.node.op.should eq Op::Copy
      r.to_a(Float32).should eq f32(1, 4, 2, 5, 3, 6)
    end

    it "raises shape errors when the op is called" do
      expect_raises(ShapeError) { a + tensor(device, [2], f32(1, 2)) }
      expect_raises(ShapeError) { tensor(device, [2, 2], f32(1, 2, 3)) }
      expect_raises(ShapeError) { a.sum(2) }
    end

    it "raises dtype errors when the op is called and when reading" do
      expect_raises(DTypeError) { a + tensor(device, [3], [1, 2, 3]) }
      expect_raises(DTypeError) { a.to_a(Int32) }
    end
  end
end
