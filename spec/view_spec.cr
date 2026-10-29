require "./spec_helper"

describe View do
  it "lays out contiguous views row-major" do
    v = View.contiguous([2, 3, 4])
    v.strides.should eq [12, 4, 1]
    v.size.should eq 24
    v.contiguous?.should be_true
  end

  it "treats rank 0 as a single element" do
    v = View.contiguous([] of Int32)
    v.size.should eq 1
    v.offset_of(0).should eq 0
  end

  it "permutes shape and strides together" do
    v = View.contiguous([2, 3, 4]).permute([2, 0, 1])
    v.shape.should eq [4, 2, 3]
    v.strides.should eq [1, 12, 4]
    v.contiguous?.should be_false
  end

  it "transposes the last two axes by default" do
    v = View.contiguous([2, 3]).transpose
    v.shape.should eq [3, 2]
    v.strides.should eq [1, 3]
    # element [1, 0] of the transpose is element [0, 1] of the original
    v.offset_of(2).should eq 1
  end

  it "rejects invalid permutations and axes" do
    expect_raises(ShapeError) { View.contiguous([2, 3]).permute([0, 0]) }
    expect_raises(ShapeError) { View.contiguous([2, 3]).transpose(0, 2) }
  end

  it "broadcasts with stride 0 on size-1 and new leading dimensions" do
    v = View.contiguous([3, 1]).expand([2, 3, 4])
    v.shape.should eq [2, 3, 4]
    v.strides.should eq [0, 1, 0]
    v.offset_of(5).should eq 1
  end

  it "rejects incompatible broadcasts" do
    expect_raises(ShapeError) { View.contiguous([3]).expand([4]) }
    expect_raises(ShapeError) { View.broadcast_shape([2, 3], [4, 3]) }
  end

  it "computes NumPy broadcast shapes" do
    View.broadcast_shape([3, 1], [4]).should eq [3, 4]
    View.broadcast_shape([] of Int32, [2, 2]).should eq [2, 2]
  end

  it "slices along an axis by moving the offset" do
    v = View.contiguous([2, 3]).slice(1, 1, 3)
    v.shape.should eq [2, 2]
    v.strides.should eq [3, 1]
    v.offset.should eq 1
    v.offset_of(2).should eq 4
    expect_raises(ShapeError) { View.contiguous([2, 3]).slice(0, 1, 3) }
  end

  it "reshapes contiguous views and refuses others" do
    View.contiguous([2, 3]).reshape([3, 2]).try(&.strides).should eq [2, 1]
    View.contiguous([2, 3]).transpose.reshape([6]).should be_nil
    expect_raises(ShapeError) { View.contiguous([2, 3]).reshape([4]) }
  end

  it "reshapes strided views without a copy where strides allow" do
    rows = View.contiguous([2, 6]).slice(1, 2, 6)
    split = rows.reshape([2, 2, 2]).should_not be_nil
    split.strides.should eq [6, 2, 1]
    split.offset.should eq 2
    rows.reshape([8]).should be_nil
    heads = View.contiguous([1, 1, 4, 3]).permute([0, 2, 1, 3])
    heads.reshape([1, 12]).try(&.strides).should eq [12, 1]
    View.contiguous([1, 3]).expand([2, 3]).reshape([2, 1, 3]).try(&.strides).should eq [0, 3, 1]
  end

  it "reads the same positions after a reshape as before" do
    rng = Random.new(7)
    200.times do
      shape = Array.new(rng.rand(1..4)) { rng.rand(1..4) }
      view = View.contiguous(shape).permute((0...shape.size).to_a.shuffle(rng))
      axis = rng.rand(view.ndim)
      start = rng.rand(view.shape[axis])
      view = view.slice(axis, start, rng.rand(start + 1..view.shape[axis]))
      target = view.shape.reject(1).flat_map { |dim| dim == 4 && rng.next_bool ? [2, 2] : [dim] }
      target.insert(rng.rand(target.size + 1), 1) if rng.next_bool
      if reshaped = view.reshape(target)
        reshaped.shape.should eq target
        view.size.times { |i| reshaped.offset_of(i).should eq view.offset_of(i) }
      end
    end
  end
end
