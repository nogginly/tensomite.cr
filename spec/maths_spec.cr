require "./spec_helper"

# Returns how many representable Float32 values lie between *a* and *b*.
private def ulps(a : Float32, b : Float32) : Int64
  ordered = ->(x : Float32) { (i = x.unsafe_as(Int32).to_i64) < 0 ? Int64.new(Int32::MIN) - i : i }
  (ordered.call(a) - ordered.call(b)).abs
end

# Returns the largest error in ulps of *fast* against *reference* (in Float64,
# then rounded) over *xs*.
private def worst(xs : Enumerable(Float32), fast : Float32 -> Float32, reference : Float64 -> Float64) : Int64
  xs.max_of { |x| ulps(fast.call(x), reference.call(x.to_f64).to_f32) }
end

private def sweep(from : Float64, to : Float64, count : Int32) : Array(Float32)
  Array.new(count) { |i| (from + (to - from) * i / (count - 1)).to_f32 }
end

# Returns *count* values spread evenly in log space from *from* to *to*.
private def log_sweep(from : Float64, to : Float64, count : Int32) : Array(Float32)
  Array.new(count) { |i| Math.exp(Math.log(from) + (Math.log(to) - Math.log(from)) * i / (count - 1)).to_f32 }
end

describe Maths do
  it "computes exp within 2 ulp" do
    worst(sweep(-87.0, 88.7, 200_001), ->Maths.exp(Float32), ->Math.exp(Float64)).should be <= 2
    worst(sweep(-1.0, 1.0, 20_001), ->Maths.exp(Float32), ->Math.exp(Float64)).should be <= 2
  end

  it "computes log within 2 ulp" do
    worst(log_sweep(1.2e-38, 3.0e38, 200_001), ->Maths.log(Float32), ->Math.log(Float64)).should be <= 2
    worst(sweep(0.5, 2.0, 20_001), ->Maths.log(Float32), ->Math.log(Float64)).should be <= 2
    worst(log_sweep(1.0e-45, 1.1e-38, 2_001), ->Maths.log(Float32), ->Math.log(Float64)).should be <= 2
  end

  it "computes tanh within 2 ulp" do
    worst(sweep(-12.0, 12.0, 200_001), ->Maths.tanh(Float32), ->Math.tanh(Float64)).should be <= 2
    worst(log_sweep(1.0e-30, 0.7, 20_001), ->Maths.tanh(Float32), ->Math.tanh(Float64)).should be <= 2
  end

  it "handles special values" do
    inf, nan = Float32::INFINITY, Float32::NAN
    {Maths.exp(-inf), Maths.exp(inf), Maths.exp(0_f32), Maths.exp(-100_f32), Maths.exp(100_f32)}.should eq({0_f32, inf, 1_f32, 0_f32, inf})
    {Maths.log(1_f32), Maths.log(0_f32), Maths.log(inf)}.should eq({0_f32, -inf, inf})
    Maths.log(-1_f32).nan?.should be_true
    {Maths.tanh(inf), Maths.tanh(-inf), Maths.tanh(0_f32), Maths.tanh(30_f32)}.should eq({1_f32, -1_f32, 0_f32, 1_f32})
    Maths.tanh(-0_f32).sign_bit.should eq -1
    [Maths.exp(nan), Maths.log(nan), Maths.tanh(nan)].all?(&.nan?).should be_true
  end
end
