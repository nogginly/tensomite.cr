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

  it "raises for a target out of range that is not ignored, when the loss is realised" do
    loss = logits.cross_entropy(Tensor.from([2, 5, 1], [3]), ignore_index: -1)
    expect_raises(ExecutionError, /index 5 out of range for 3 columns/) { loss.realize }
    expect_raises(ExecutionError, /index -1 out of range/) { logits.cross_entropy(Tensor.from([2, -1, 1], [3])).realize }
  end

  it "reads no target on the host: building the loss realises nothing" do
    lazy = Tensor.from([1, 0, 0], [3]) + Tensor.from([1, 0, 1], [3])
    loss = logits.cross_entropy(lazy, ignore_index: -1)
    lazy.node.realized?.should be_false
    loss.to_a(Float32).should eq logits.cross_entropy(targets).to_a(Float32)
  end

  it "gives the bits, value and gradient, of picking each row's target by a flat gather" do
    DEVICES.each do |device|
      ids = [3, 49, 0, 17, 17, 8, 22]
      x = Tensor.from(random_values(7 * 50, 3), [7, 50], device)
      flat = Tensor.from(ids.map_with_index { |id, row| row * 50 + id }, [7], device)
      gathered = -x.log_softmax.reshape([350, 1]).gather(flat).mean
      picked = x.cross_entropy(Tensor.from(ids, [7], device))
      grads = {Tensomite.grad(picked, [x]).first, Tensomite.grad(gathered, [x]).first}
      picked.to_a(Float32).should eq gathered.to_a(Float32)
      grads[0].to_a(Float32).should eq grads[1].to_a(Float32)
    end
  end
end

describe "Tensor#pick and #place" do
  it "picks each row's element at its id's column, 0 where the id is ignored" do
    DEVICES.each do |device|
      table = tensor(device, [3, 2], f32(1, 2, 3, 4, 5, 6)).transpose
      table.pick(Tensor.from([2, -1], [2], device), ignore: -1).to_a(Float32).should eq f32(5, 0)
      table.pick(Tensor.from([0, 2], [2], device)).to_a(Float32).should eq f32(1, 6)
    end
  end

  it "places each row's value at its id's column, the adjoint of pick" do
    DEVICES.each do |device|
      placed = tensor(device, [3], f32(1, 2, 3)).place(Tensor.from([2, -1, 0], [3], device), 3, ignore: -1)
      placed.to_a(Float32).should eq f32(0, 0, 1, 0, 0, 0, 3, 0, 0)
    end
  end

  it "has place as its gradient" do
    check_gradients([[3, 4]]) { |(x)| x.pick(Tensor.from([1, -1, 3], [3], x.device), ignore: -1) }
  end

  it "checks its ids' dtype and shape" do
    table = Tensor.from(f32(1, 2, 3, 4), [2, 2])
    expect_raises(DTypeError, /I32/) { table.pick(Tensor.from(f32(0, 1), [2])) }
    expect_raises(ShapeError, /one id per row/) { table.pick(Tensor.from([0, 1, 0], [3])) }
    expect_raises(ExecutionError, /index 2 out of range for 2 columns/) { table.pick(Tensor.from([0, 2], [2])).realize }
  end
end
