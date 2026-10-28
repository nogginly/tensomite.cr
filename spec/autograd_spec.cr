require "./spec_helper"

describe "Tensomite.grad" do
  it "differentiates the binary ops, broadcasting included" do
    check_gradients([[2, 3], [3]]) { |(a, b)| a + b }
    check_gradients([[2, 3], [2, 1]]) { |(a, b)| a - b }
    check_gradients([[2, 3], [2, 3]]) { |(a, b)| a * b }
    check_gradients([[2, 3], [3]]) { |(a, b)| a / (b * b + 1) }
    check_gradients([[2, 3], [2, 3]]) { |(a, b)| a.maximum(b) }
  end

  it "differentiates the unary ops" do
    check_gradients([[2, 3]], &.first.exp)
    check_gradients([[2, 3]]) { |(a)| (a * a + 1).log }
    check_gradients([[2, 3]], &.first.tanh)
    check_gradients([[2, 3]]) { |(a)| (a * a + 1).sqrt }
  end

  it "differentiates reductions over an axis and over everything" do
    check_gradients([[2, 3, 4]], &.first.sum(1))
    check_gradients([[2, 3]], &.first.sum)
    check_gradients([[2, 3, 4]], &.first.max(-1))
    check_gradients([[2, 3]], &.first.max)
    check_gradients([[2, 3]], &.first.mean(0, keepdims: true))
  end

  it "splits the gradient of max evenly between ties" do
    a = Tensor.from(f32(1, 3, 3, 2), [4])
    Tensomite.grad(a.max, [a]).first.to_a(Float32).should eq f32(0, 0.5, 0.5, 0)
  end

  it "differentiates views" do
    check_gradients([[2, 3]]) { |(a)| a.transpose.contiguous }
    check_gradients([[2, 4]]) { |(a)| a.slice(1, 1, 3) * 2 }
    check_gradients([[3]]) { |(a)| a.reshape([1, 3]).expand([2, 3]) * a }
    check_gradients([[2, 3]]) { |(a)| a.permute([1, 0]).reshape([6]) }
  end

  it "differentiates cat, gather and matmul" do
    check_gradients([[2, 2], [1, 2]]) { |(a, b)| Tensor.cat([a, b, a], 0) }
    check_gradients([[4, 3]]) { |(a)| a.gather(Tensor.from([2, 0, 2], [3], a.device)) }
    check_gradients([[2, 3], [3, 4]]) { |(a, b)| a.matmul(b) }
    check_gradients([[2, 3], [3, 4], [4]]) { |(a, b, c)| a.matmul(b, c) }
    check_gradients([[2, 2, 3], [3, 2]]) { |(a, b)| a.matmul(b) }
  end

  it "differentiates the composites" do
    check_gradients([[2, 4]], &.first.softmax)
    check_gradients([[2, 4]], &.first.log_softmax)
    check_gradients([[2, 4]], &.first.gelu)
    check_gradients([[2, 4], [4], [4]]) { |(x, w, b)| x.layernorm(w, b) }
    check_gradients([[3, 4]]) { |(x)| x.cross_entropy(Tensor.from([1, 3, 0], [3], x.device)) }
  end

  it "accumulates gradients from every use of a tensor" do
    a = Tensor.from(f32(1, 2), [2])
    Tensomite.grad((a * a + a).sum, [a]).first.to_a(Float32).should eq f32(3, 5)
  end

  it "gives zeros for a param the loss does not use" do
    a = Tensor.from(f32(1, 2), [2])
    b = Tensor.from(f32(3), [1])
    Tensomite.grad(a.sum, [a, b]).map(&.to_a(Float32)).should eq [f32(1, 1), f32(0)]
  end

  it "treats detached tensors as constants" do
    a = Tensor.from(f32(1, 2), [2])
    c = (a * 3).detach
    Tensomite.grad((a * c).sum, [a]).first.to_a(Float32).should eq f32(3, 6)
  end

  it "raises for a realised intermediate" do
    a = Tensor.from(f32(1, 2), [2])
    b = (a * a).realize
    expect_raises(Error, /realised/) { Tensomite.grad(b.sum, [a]) }
  end

  it "raises for a loss with more than one element" do
    a = Tensor.from(f32(1, 2), [2])
    expect_raises(ShapeError) { Tensomite.grad(a * 2, [a]) }
  end
end
