require "./spec_helper"
require "json"

private def hex_bytes(hex : String) : Bytes
  Bytes.new(hex.size // 2) { |i| hex[2 * i, 2].to_u8(16) }
end

describe Quant do
  expected = JSON.parse(File.read("#{__DIR__}/fixtures/quant/expected.json"))
  rows = expected["rows"].as_a.map(&.as_a.map(&.as_f.to_f32))

  {DType::Q8_0, DType::Q4_0, DType::Q4_1}.each do |dtype|
    reference = expected[dtype.to_s]

    it "quantises to #{dtype} as ggml's reference does, byte for byte" do
      rows.each_with_index do |row, i|
        Quant.quantize(dtype, Slice.new(row.to_unsafe, row.size)).should eq hex_bytes(reference["bytes"][i].as_s)
      end
    end

    it "dequantises #{dtype} as ggml does" do
      rows.each_with_index do |row, i|
        values = Slice(Float32).new(row.size)
        Quant.dequantize(dtype, hex_bytes(reference["bytes"][i].as_s), values)
        values.to_a.should eq reference["values"][i].as_a.map(&.as_f.to_f32)
      end
    end
  end

  {DType::Q8_0, DType::Q4_0, DType::Q4_1}.each do |dtype|
    it "multiplies #{dtype} weights as the reference does, bit for bit" do
      rng = Random.new(11)
      [{1, 37, 64}, {3, 5, 1280}, {2, 9, 4096}].each do |(count, n, k)|
        values = Array.new(n * k) { rng.rand(-2.0..2.0).to_f32 }
        values[0, 32].fill(0f32)
        values[32 * 3] = 50f32
        x = Array.new(count * k) { rng.rand(-3.0..3.0).to_f32 }
        x[k, 32].fill(0f32) if count > 1
        weights = Quant.quantize(dtype, Slice.new(values.to_unsafe, values.size))
        fast, reference = Slice(Float32).new(count * n), Slice(Float32).new(count * n)
        xs = Slice.new(x.to_unsafe, x.size)
        Quant.matmul(dtype, xs, weights, count, n, k, fast)
        Quant.matmul_reference(dtype, xs, weights, count, n, k, reference)
        fast.to_a.map(&.unsafe_as(UInt32)).should eq reference.to_a.map(&.unsafe_as(UInt32))
      end
    end
  end

  it "sizes blocks" do
    DType::Q8_0.bytes_for(64).should eq 68
    DType::Q4_0.bytes_for(64).should eq 36
    DType::Q4_1.bytes_for(32).should eq 20
    expect_raises(ShapeError) { DType::Q4_0.bytes_for(48) }
    DType::Q4_0.block?.should be_true
    DType::F32.block?.should be_false
  end
end

describe "Block-format tensors" do
  expected = JSON.parse(File.read("#{__DIR__}/fixtures/quant/expected.json"))
  table = expected["table"].as_a.flat_map(&.as_a.map(&.as_f.to_f32))

  (DEVICES + GPUS).each do |device|
    it "quantise and dequantise on #{device.class}" do
      x = Tensor.from(table, [3, 64], device)
      {DType::Q8_0, DType::Q4_0, DType::Q4_1}.each do |dtype|
        q = x.quantize(dtype)
        q.dtype.should eq dtype
        q.shape.should eq [3, 64]
        q.device.should be device
        q.dequantize.to_a(Float32).should eq expected[dtype.to_s]["table"].as_a.flat_map(&.as_a.map(&.as_f.to_f32))
      end
    end

    it "read from GGUF on #{device.class}" do
      file = GGUF.open("#{__DIR__}/fixtures/quant/blocks.gguf")
      {DType::Q8_0, DType::Q4_0, DType::Q4_1}.each do |dtype|
        q = file[dtype.to_s.downcase, device]
        q.dtype.should eq dtype
        q.shape.should eq [3, 64]
        q.dequantize.to_a(Float32).should eq expected[dtype.to_s]["table"].as_a.flat_map(&.as_a.map(&.as_f.to_f32))
      end
    end
  end

  it "slice by whole rows, move between devices, and keep their bytes" do
    q = Tensor.from(table, [3, 64]).quantize(DType::Q4_0)
    whole = q.dequantize.to_a(Float32)
    q.slice(0, 1, 3).dequantize.to_a(Float32).should eq whole[64..]
    q.to(SimDiscrete.default).slice(0, 2, 3).dequantize.to_a(Float32).should eq whole[128..]
  end

  it "stay within half a step of the values they came from, plus the scale's F16 rounding" do
    x = Tensor.from(table, [3, 64])
    {DType::Q8_0, DType::Q4_0, DType::Q4_1}.each do |dtype|
      got = x.quantize(dtype).dequantize.to_a(Float32)
      table.each_slice(32).with_index do |block, b|
        amax, range = block.max_of(&.abs), block.max - block.min
        bound = case dtype
                when .q8_0? then amax / 254
                when .q4_0? then amax / 8
                else             range / 30
                end
        block.each_with_index { |want, j| got[b * 32 + j].should be_close(want, bound + amax / 1024) }
      end
    end
  end

  it "refuse views across blocks, ops, and reading as F32" do
    q = Tensor.from(table, [3, 64]).quantize(DType::Q8_0)
    expect_raises(DTypeError, /whole rows/) { q.transpose }
    expect_raises(DTypeError, /whole rows/) { q.slice(1, 0, 32) }
    expect_raises(DTypeError, /whole rows/) { q.reshape([192]) }
    expect_raises(DTypeError, /Q8_0/) { q + q }
    expect_raises(DTypeError, /Q8_0/) { q.to_a(Float32) }
    expect_raises(ShapeError) { Tensor.from(Array.new(48, 1.0_f32), [48]).quantize(DType::Q4_0) }
    expect_raises(DTypeError) { Tensor.from(table, [3, 64]).quantize(DType::F32) }
  end
end
