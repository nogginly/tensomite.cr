require "./spec_helper"

# Tensors of more than 2 GiB, which need as much free memory again: run with
# TENSOMITE_BIG_SPECS=1.
ROWS_3GIB = 786_432 # rows of 1,024 F32: 3 GiB
Q8_ROWS   = 500_000 # rows of 4,096 Q8_0: 2.2 GB, under 2^31 elements

# Returns *rows* rows of *width* values in Q8_0 blocks, each value 1: every
# block's scale is 1.0 and its 32 quantised values are 1. A product with a
# row of ones gives 4,095.75 an output, not 4,096: the ones, quantised to 127
# with an F16 scale of 1/127, come back as 0.99994.
def q8_0_ones(rows : Int32, width : Int32) : Memory
  blocks = rows.to_i64 * (width // DType.block_size)
  memory = Memory.allocate(DType::Q8_0.bytes_for(rows.to_i64 * width)).fill(1)
  bytes = memory.pointer
  blocks.times do |block|
    at = bytes + block * DType::Q8_0.block_bytes
    at[0] = 0x00_u8
    at[1] = 0x3c_u8
  end
  memory
end

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

    it "is saved and loaded as safetensors and as GGUF" do
      GC.collect
      big = Tensor.scalar(0, DType::F32).expand([ROWS_3GIB, 1024]).contiguous
      written = big.write(Tensor.full([1, 1024], 2), 0, ROWS_3GIB - 1).realize
      {"safetensors", "gguf"}.each do |format|
        path = File.tempname("tensomite", ".#{format}")
        begin
          if format == "gguf"
            GGUF.write(path, {"big" => written})
            loaded = GGUF.open(path)["big"]
          else
            SafeTensors.write(path, {"big" => written})
            loaded = SafeTensors.open(path)["big"]
          end
          loaded.shape.should eq [ROWS_3GIB, 1024]
          loaded.sum.to_a(Float32).should eq [2048f32]
          loaded.slice(0, ROWS_3GIB - 1, ROWS_3GIB).to_a(Float32).should eq Array.new(1024, 2f32)
        ensure
          File.delete?(path)
        end
        GC.collect
      end
    end

    (DEVICES + GPUS).each do |device|
      it "multiplies and gathers a Q8_0 weight of more than 2 GiB on #{device.class}" do
        GC.collect
        weight = Tensor.from_blocks(q8_0_ones(Q8_ROWS, 4096), [Q8_ROWS, 4096], DType::Q8_0, device)
        weight.realize.node.bytesize.should be > Int32::MAX
        outputs = Tensor.full([1, 4096], 1, device: device).linear(weight)
        outputs.slice(1, Q8_ROWS - 1, Q8_ROWS).to_a(Float32).first.should be_close(4096f32, 0.5)
        weight.gather(Tensor.from([Q8_ROWS - 1], [1], device)).to_a(Float32).should eq Array.new(4096, 1f32)
        GC.collect
      end
    end

    it "saves and loads a Q8_0 weight of more than 2 GiB as GGUF" do
      GC.collect
      path = File.tempname("tensomite", ".gguf")
      begin
        GGUF.write(path, {"w" => Tensor.from_blocks(q8_0_ones(Q8_ROWS, 4096), [Q8_ROWS, 4096], DType::Q8_0)})
        loaded = GGUF.open(path)["w"]
        loaded.dtype.should eq DType::Q8_0
        loaded.gather(Tensor.from([Q8_ROWS - 1], [1])).to_a(Float32).should eq Array.new(4096, 1f32)
      ensure
        File.delete?(path)
      end
    end
  else
    pending "needs TENSOMITE_BIG_SPECS=1 and about 7 GB of free memory"
  end
end
