require "./spec_helper"

DEVICES.each do |device|
  describe "Structural ops on #{device.class}" do
    # a = [[1, 2, 3], [4, 5, 6]], b = [[6, 5, 4], [3, 2, 1]]
    a = tensor(device, [2, 3], f32(1, 2, 3, 4, 5, 6))
    b = tensor(device, [2, 3], f32(6, 5, 4, 3, 2, 1))
    # w = [[1, 2], [3, 4], [5, 6]]
    w = tensor(device, [3, 2], f32(1, 2, 3, 4, 5, 6))

    describe "slice" do
      it "selects a range along an axis without copying" do
        s = a.slice(1, 1, 3)
        s.node.should be a.node
        s.to_a(Float32).should eq f32(2, 3, 5, 6)
        a.slice(0, 1, 2).to_a(Float32).should eq f32(4, 5, 6)
        (a.slice(1, 0, 2) + 1).to_a(Float32).should eq f32(2, 3, 5, 6)
      end

      it "rejects ranges outside the axis" do
        expect_raises(ShapeError) { a.slice(1, 2, 4) }
      end
    end

    describe "cat" do
      it "joins along either axis" do
        Tensor.cat([a, b], 0).to_a(Float32).should eq f32(1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1)
        Tensor.cat([a, b], 1).to_a(Float32).should eq f32(1, 2, 3, 6, 5, 4, 4, 5, 6, 3, 2, 1)
      end

      it "joins transposed views" do
        # a.T = [[1, 4], [2, 5], [3, 6]], b.T = [[6, 3], [5, 2], [4, 1]]
        c = Tensor.cat([a.transpose, b.transpose], 1)
        c.shape.should eq [3, 4]
        c.to_a(Float32).should eq f32(1, 4, 6, 3, 2, 5, 5, 2, 3, 6, 4, 1)
      end

      it "rejects mismatched shapes and dtypes" do
        expect_raises(ShapeError) { Tensor.cat([a, w], 0) }
        expect_raises(DTypeError) { Tensor.cat([a, tensor(device, [1, 3], [1, 2, 3])], 0) }
      end
    end

    describe "write" do
      it "writes into its target in place, keeping the rest" do
        target = tensor(device, [3, 2], f32(1, 2, 3, 4, 5, 6))
        written = target.write(tensor(device, [1, 2], f32(9, 8)), 0, 1).realize
        written.to_a(Float32).should eq f32(1, 2, 9, 8, 5, 6)
        written.node.buffer!.should be target.node.buffer!
        written.write(tensor(device, [2, 3], f32(7, 7, 7, 0, 0, 0)).transpose.slice(1, 0, 1), 1, 1).to_a(Float32).should eq f32(1, 7, 9, 7, 5, 7)
      end

      it "rejects mismatched shapes, ranges past the end and views of part of a buffer" do
        target = tensor(device, [3, 2], f32(1, 2, 3, 4, 5, 6))
        expect_raises(ShapeError) { target.write(tensor(device, [1, 3], f32(1, 2, 3)), 0, 0) }
        expect_raises(ShapeError) { target.write(tensor(device, [2, 2], f32(1, 2, 3, 4)), 0, 2) }
        expect_raises(ShapeError) { target.slice(0, 1, 3).write(tensor(device, [1, 2], f32(1, 2)), 0, 0) }
      end
    end

    describe "gather" do
      it "selects table rows by id" do
        ids = tensor(device, [3], [2, 0, 2])
        w.gather(ids).to_a(Float32).should eq f32(5, 6, 1, 2, 5, 6)
      end

      it "keeps the shape of the ids" do
        g = w.gather(tensor(device, [2, 2], [0, 1, 1, 0]))
        g.shape.should eq [2, 2, 2]
        g.to_a(Float32).should eq f32(1, 2, 3, 4, 3, 4, 1, 2)
      end

      it "reads a transposed table through its view" do
        # w.T = [[1, 3, 5], [2, 4, 6]]
        w.transpose.gather(tensor(device, [1], [1])).to_a(Float32).should eq f32(2, 4, 6)
      end

      it "raises for an id out of range when realised, not when called" do
        g = w.gather(tensor(device, [1], [3]))
        expect_raises(ExecutionError, /index 3 out of range for 3 rows/) { g.realize }
      end

      it "rejects non-I32 ids when called" do
        expect_raises(DTypeError) { w.gather(tensor(device, [1], f32(0))) }
      end
    end

    describe "matmul" do
      it "multiplies matrices" do
        # a @ w = [[1 + 6 + 15, 2 + 8 + 18], [4 + 15 + 30, 8 + 20 + 36]]
        a.matmul(w).to_a(Float32).should eq f32(22, 28, 49, 64)
      end

      it "multiplies through a transposed view" do
        # a @ a.T = [[1 + 4 + 9, 4 + 10 + 18], [4 + 10 + 18, 16 + 25 + 36]]
        a.matmul(a.transpose).to_a(Float32).should eq f32(14, 32, 32, 77)
      end

      it "broadcasts a batch against a single matrix" do
        batch = Tensor.cat([a.reshape([1, 2, 3]), b.reshape([1, 2, 3])], 0)
        # b @ w = [[6 + 15 + 20, 12 + 20 + 24], [3 + 6 + 5, 6 + 8 + 6]]
        r = batch.matmul(w)
        r.shape.should eq [2, 2, 2]
        r.to_a(Float32).should eq f32(22, 28, 49, 64, 41, 56, 14, 20)
      end

      it "adds a bias with the bits of a separate add, batched and in I32" do
        bias = tensor(device, [2], f32(0.1, -2.5))
        a.matmul(w, bias).to_a(Float32).should eq (a.matmul(w) + bias).to_a(Float32)
        batch = Tensor.cat([a.reshape([1, 2, 3]), b.reshape([1, 2, 3])], 0)
        rows = tensor(device, [2, 1], f32(3, 0.25))
        batch.matmul(w, rows).to_a(Float32).should eq (batch.matmul(w) + rows).to_a(Float32)
        ints = tensor(device, [1, 2], [1, 2])
        ints.matmul(tensor(device, [2, 1], [3, 4]), tensor(device, [1], [5])).to_a(Int32).should eq [16]
        expect_raises(ShapeError) { a.matmul(w, tensor(device, [3], f32(1, 2, 3))) }
      end

      it "multiplies I32 matrices" do
        tensor(device, [1, 2], [1, 2]).matmul(tensor(device, [2, 1], [3, 4])).to_a(Int32).should eq [11]
      end

      it "rejects mismatched inner dimensions" do
        expect_raises(ShapeError) { a.matmul(a) }
      end
    end
  end
end
