require "./spec_helper"

# Returns a [rows, cols] tensor of seeded random values in [-1, 1) on *device*.
private def random(device : Device, rows : Int32, cols : Int32, seed : Int32) : Tensor
  rng = Random.new(seed)
  Tensor.from(Array(Float32).new(rows * cols) { rng.rand(-1f32...1f32) }, [rows, cols], device)
end

# Returns the largest elementwise difference relative to the largest reference value.
private def relative_error(actual : Array(Float32), expected : Array(Float32)) : Float64
  scale = expected.max_of(&.abs).to_f64
  actual.zip(expected).max_of { |x, y| (x - y).abs.to_f64 } / scale
end

describe BLAS do
  blas = BLAS.default
  cpu = CPU.default

  describe "M3 exit: sgemm within 1e-5 relative error of the naive matmul" do
    # Pairs build the same product on both devices from the same seeds.
    cases = {
      "contiguous"       => ->(d : Device) { random(d, 33, 47, 1).matmul(random(d, 47, 29, 2)) },
      "transposed left"  => ->(d : Device) { random(d, 47, 33, 1).transpose.matmul(random(d, 47, 29, 2)) },
      "transposed right" => ->(d : Device) { random(d, 33, 47, 1).matmul(random(d, 29, 47, 2).transpose) },
      "sliced columns"   => ->(d : Device) { random(d, 33, 60, 1).slice(1, 5, 52).matmul(random(d, 47, 29, 2)) },
      "broadcast batch"  => ->(d : Device) { random(d, 66, 47, 1).reshape([2, 33, 47]).matmul(random(d, 47, 29, 2)) },
      "broadcast rows"   => ->(d : Device) { random(d, 1, 47, 1).expand([33, 47]).matmul(random(d, 47, 29, 2)) },
      "vector by matrix" => ->(d : Device) { random(d, 1, 47, 1).matmul(random(d, 47, 29, 2)) },
      "matrix by column" => ->(d : Device) { random(d, 33, 47, 1).matmul(random(d, 47, 1, 2)) },
    }

    cases.each do |name, build|
      it "matches for #{name}" do
        expected = build.call(cpu)
        actual = build.call(blas)
        actual.shape.should eq expected.shape
        relative_error(actual.to_a(Float32), expected.to_a(Float32)).should be <= 1e-5
      end
    end
  end

  it "fills with zeros when the inner dimension is empty" do
    a = Tensor.from([] of Float32, [2, 0], blas)
    b = Tensor.from([] of Float32, [0, 3], blas)
    a.matmul(b).to_a(Float32).should eq f32(0, 0, 0, 0, 0, 0)
  end

  it "gives the same bits on every run" do
    product = -> { random(blas, 64, 64, 3).matmul(random(blas, 64, 64, 4)).to_a(Float32) }
    product.call.should eq product.call
  end

  it "uses the host kernel for I32" do
    a = Tensor.from([1, 2, 3, 4], [2, 2], blas)
    a.matmul(a).to_a(Int32).should eq [7, 10, 15, 22]
  end

  it "reports that it is not shape-invariant, unlike host devices" do
    blas.shape_invariant?.should be_false
    cpu.shape_invariant?.should be_true
    SimDiscrete.default.shape_invariant?.should be_true
  end
end
