# Writes inputs and the host's results for scripts/emulate_metal/blocks.sh: for
# each block format, rows of activations, a weight, a bias, and the host's
# `x.linear(w, bias)`; then a table, ids, and the host's gather.
require "../../src/tensomite"
include Tensomite

private def floats(io : IO, values : Array(Float32)) : Nil
  io.write_bytes(values.size.to_u32, IO::ByteFormat::LittleEndian)
  values.each { |value| io.write_bytes(value, IO::ByteFormat::LittleEndian) }
end

private def raw(io : IO, bytes : Bytes) : Nil
  io.write_bytes(bytes.size.to_u32, IO::ByteFormat::LittleEndian)
  io.write(bytes)
end

private def values(count : Int32, seed : Int32) : Array(Float32)
  rng = Random.new(seed)
  Array.new(count) { rng.rand(-2.0..2.0).to_f32 }
end

rows, n, k = 5, 37, 1280
File.open(ARGV[0], "w") do |io|
  {DType::Q8_0, DType::Q4_0, DType::Q4_1}.each do |dtype|
    x, weights, bias = values(rows * k, 1), values(n * k, 2), values(n, 3)
    w = Tensor.from(weights, [n, k]).quantize(dtype)
    io.write_bytes(dtype.value.to_u32 - DType::Q8_0.value.to_u32, IO::ByteFormat::LittleEndian)
    {rows, n, k}.each { |dim| io.write_bytes(dim.to_u32, IO::ByteFormat::LittleEndian) }
    floats(io, x)
    raw(io, Quant.quantize(dtype, Slice.new(weights.to_unsafe, weights.size)))
    floats(io, bias)
    floats(io, Tensor.from(x, [rows, k]).linear(w, Tensor.from(bias, [n])).to_a(Float32))
    ids = [3, 0, 36, 3]
    floats(io, w.gather(Tensor.from(ids, [4])).to_a(Float32))
  end
end
