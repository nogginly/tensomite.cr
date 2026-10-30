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
end
