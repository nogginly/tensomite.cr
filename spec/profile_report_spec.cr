require "./spec_helper"

describe Tensomite::ProfileReport do
  totals = ->(entries : Array({String, Int32, Float64})) do
    Tensomite::ProfileReport::Totals.new({0, 0.0}).tap { |hash| entries.each { |(name, count, seconds)| hash[name] = {count, seconds} } }
  end

  report = Tensomite::ProfileReport.new(
    "GPU", "step", 2, 0.010,
    [{"launching", 0.002}, {"waiting", 0.004}],
    ["launches: 7"],
    totals.call([{"Sum F32", 2, 0.001}]),
    totals.call([
      {"MatMul [1,4]·[4,8]", 4, 0.000002},
      {"MatMul [2,4]·[4,8]", 2, 0.000004},
      {"Add [2,8]", 2, 0.000002},
    ]),
    top: 2,
  ).to_s

  it "splits the wall time into its parts, host ops and the rest, per run" do
    report.should contain "wall|5.000\n"
    report.should contain "launching|1.000\nwaiting|2.000\nhost ops|0.500\neverything else|1.500\n"
    report.should contain "launches: 7\nhost op Sum F32: 1.0 each, 0.500 ms\n"
  end

  it "sums GPU time by op, largest first" do
    report.should contain "total 0.004 ms, 0% of the first run's wall time"
    report.should contain "MatMul|3.0|0.003|75%\nAdd|1.0|0.001|25%\n"
  end

  it "lists the largest by shape, a one-row matmul in GB/s and others in GFLOP/s" do
    report.should contain "the 2 largest"
    report.should contain "MatMul [1,4]·[4,8]|2.0|0.001|25%|0.3 GB/s\n"
    report.should contain "MatMul [2,4]·[4,8]|1.0|0.002|50%|0.1 GFLOP/s\n"
    report.should_not contain "Add [2,8]|"
  end

  it "gives a quantised product its weight's bytes read per second" do
    # 2,048 × 1,024 Q4_0 values are 65,536 blocks of 18 bytes: 1,179,648 bytes,
    # read 3 times in 0.001 s over 1 run.
    quantised = Tensomite::ProfileReport.new("GPU", "token", 1, 0.010, [] of {String, Float64}, [] of String,
      totals.call([] of {String, Int32, Float64}), totals.call([{"BlockMatMul Q4_0 [1,1024]·[2048,1024]", 3, 0.001}])).to_s
    quantised.should contain "BlockMatMul Q4_0 [1,1024]·[2048,1024]|3.0|1.000|100%|3.5 GB/s\n"
  end

  it "labels a quantised product with its weight's format and both shapes" do
    w = Tensor.from(Array.new(64 * 32) { |i| (i % 7).to_f32 }, [64, 32]).quantize(DType::Q4_0)
    x = Tensor.zeros([2, 32])
    node = x.linear(w).node
    Tensomite::ProfileReport.label(node, shapes: true).should eq "BlockMatMul Q4_0 [2,32]·[64,32]"
    Tensomite::ProfileReport.label(node, shapes: false).should eq "BlockMatMul"
  end
end
