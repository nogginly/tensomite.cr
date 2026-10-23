require "./spec_helper"

# Returns `a @ b` added in `k` order through `View#offset_of`: the definition every
# host matmul must reproduce bit for bit.
private def reference_matmul(a : Tensor, b : Tensor) : Array(Float32)
  xs, ys = a.contiguous.to_a(Float32), b.contiguous.to_a(Float32)
  m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
  batches = a.size // (m * k)
  Array(Float32).new(batches * m * n) do |i|
    batch, rem = i.divmod(m * n)
    row, col = rem.divmod(n)
    acc = 0f32
    k.times { |step| acc += xs[(batch * m + row) * k + step] * ys[((b.size == k * n ? 0 : batch) * k + step) * n + col] }
    acc
  end
end

private def random(shape : Array(Int32), seed : Int32) : Tensor
  rng = Random.new(seed)
  Tensor.from(Array(Float32).new(shape.product) { rng.rand(-1f32...1f32) }, shape)
end

describe "Host matmul kernel" do
  it "adds in k order for every operand layout, bit for bit" do
    cases = {
      {random([5, 7], 1), random([7, 3], 2)},
      {random([7, 5], 1).transpose, random([3, 7], 2).transpose},
      {random([5, 9], 1).slice(1, 1, 8), random([7, 3], 2)},
      {random([2, 5, 7], 1), random([7, 3], 2)},
    }
    cases.each do |a, b|
      a.matmul(b).to_a(Float32).should eq reference_matmul(a, b)
    end
  end
end
