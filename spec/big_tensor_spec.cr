require "./spec_helper"

# Tensors of more than 2 GiB, which need as much free memory again: run with
# TENSOMITE_BIG_SPECS=1.
ROWS_3GIB = 786_432 # rows of 1,024 F32: 3 GiB

describe "a tensor of more than 2 GiB" do
  if ENV["TENSOMITE_BIG_SPECS"]? == "1"
    (DEVICES + GPUS).each do |device|
      it "is created, written at its last row, summed and read on #{device.class}" do
        big = Tensor.scalar(0, DType::F32, device).expand([ROWS_3GIB, 1024]).contiguous
        big.realize.node.bytesize.should eq 3_i64 << 30
        last = ROWS_3GIB - 1
        written = big.write(Tensor.full([1, 1024], 2, device: device), 0, last)
        sums = written.sum(1)
        sums.slice(0, last, ROWS_3GIB).to_a(Float32).should eq [2048f32]
        sums.slice(0, 0, 1).to_a(Float32).should eq [0f32]
        written.sum.to_a(Float32).should eq [2048f32]
        written.slice(0, last, ROWS_3GIB).to_a(Float32).should eq Array.new(1024, 2f32)
        GC.collect
      end
    end
  else
    pending "needs TENSOMITE_BIG_SPECS=1 and about 7 GB of free memory"
  end
end
