require "./spec_helper"

describe "Tensor#cross_entropy" do
  logits = Tensor.from([1, 2, 3, 0, 1, 0, -1, 2, 0.5].map(&.to_f32), [3, 3])
  targets = Tensor.from([2, 0, 1], [3])

  it "gives each row's loss with reduction: :none, the bits of that row alone" do
    losses = logits.cross_entropy(targets, reduction: :none)
    losses.shape.should eq [3]
    3.times do |row|
      alone = logits.slice(0, row, row + 1).cross_entropy(targets.slice(0, row, row + 1))
      losses.to_a(Float32)[row].should eq alone.to_a(Float32).first
    end
  end

  it "sums the rows' losses with reduction: :sum" do
    total = logits.cross_entropy(targets, reduction: :sum).to_a(Float32).first
    total.should be_close(logits.cross_entropy(targets, reduction: :none).to_a(Float32).sum, 1e-6)
  end

  it "leaves out rows whose target is ignore_index" do
    kept = Tensor.from([1, 2, 3, -1, 2, 0.5].map(&.to_f32), [2, 3]).cross_entropy(Tensor.from([2, 1], [2]))
    ignoring = logits.cross_entropy(Tensor.from([2, -1, 1], [3]), ignore_index: -1)
    ignoring.to_a(Float32).should eq kept.to_a(Float32)
    none = logits.cross_entropy(Tensor.from([2, -1, 1], [3]), reduction: :none, ignore_index: -1).to_a(Float32)
    none[1].should eq 0f32
    none[0].should eq logits.cross_entropy(targets, reduction: :none).to_a(Float32)[0]
  end

  it "gives the default's bits when no row is ignored" do
    logits.cross_entropy(targets, ignore_index: -1).to_a(Float32).should eq logits.cross_entropy(targets).to_a(Float32)
  end

  it "gives NaN as the mean, and 0 as the sum, of no rows" do
    all = Tensor.from([-1, -1, -1], [3])
    logits.cross_entropy(all, ignore_index: -1).to_a(Float32).first.nan?.should be_true
    logits.cross_entropy(all, reduction: :sum, ignore_index: -1).to_a(Float32).should eq [0f32]
  end

  it "raises for a target out of range that is not ignored" do
    expect_raises(ShapeError, /out of range/) { logits.cross_entropy(Tensor.from([2, 5, 1], [3]), ignore_index: -1) }
  end
end
