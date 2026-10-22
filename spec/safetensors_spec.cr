require "./spec_helper"
require "json"

# Writes a safetensors file holding *tensors* (name to dtype, shape and raw bytes)
# and returns its path.
private def write_safetensors(tensors : Hash(String, {String, Array(Int32), Bytes})) : String
  offset = 0
  meta = tensors.transform_values do |(dtype, shape, bytes)|
    entry = {"dtype" => JSON::Any.new(dtype), "shape" => JSON::Any.new(shape.map { |dim| JSON::Any.new(dim.to_i64) }),
             "data_offsets" => JSON::Any.new([JSON::Any.new(offset.to_i64), JSON::Any.new((offset + bytes.size).to_i64)])}
    offset += bytes.size
    JSON::Any.new(entry)
  end
  json = meta.to_json
  json += " " * ((8 - json.bytesize % 8) % 8)
  path = File.tempname("tensomite", ".safetensors")
  File.open(path, "w") do |file|
    file.write_bytes(json.bytesize.to_u64, IO::ByteFormat::LittleEndian)
    file << json
    tensors.each_value { |(_, _, bytes)| file.write(bytes) }
  end
  path
end

private def f32_bytes(*values : Number) : Bytes
  Slice.new(values.size) { |i| values[i].to_f32 }.to_unsafe_bytes.dup
end

describe SafeTensors do
  path = write_safetensors({
    "a" => {"F32", [2, 2], f32_bytes(1, 2, 3, 4)},
    "b" => {"I32", [3], Slice[7, 8, 9].to_unsafe_bytes.dup},
  })
  weights = SafeTensors.open(path)

  it "lists tensors with their dtypes and shapes" do
    weights.names.should eq ["a", "b"]
    weights.entries["a"].shape.should eq [2, 2]
    weights.entries["b"].dtype.should eq DType::I32
  end

  it "reads tensors on every device" do
    DEVICES.each do |device|
      weights["a", device].to_a(Float32).should eq f32(1, 2, 3, 4)
      weights["b", device].to_a(Int32).should eq [7, 8, 9]
      weights["a", device].transpose.to_a(Float32).should eq f32(1, 3, 2, 4)
    end
  end

  it "maps host tensors without copying" do
    a = weights["a"]
    first = CPU.default.map(a.node.buffer!).not_nil!("CPU memory is host-visible")
    second = CPU.default.map(weights["a"].node.buffer!).not_nil!("CPU memory is host-visible")
    first.to_unsafe.should eq second.to_unsafe
  end

  it "raises for a missing tensor" do
    expect_raises(KeyError) { weights["missing"] }
  end

  it "rejects unsupported dtypes" do
    bad = write_safetensors({"h" => {"F16", [2], Bytes.new(4)}})
    expect_raises(DTypeError, /F16/) { SafeTensors.open(bad) }
  end

  it "rejects byte ranges that do not match the shape" do
    bad = write_safetensors({"a" => {"F32", [3], f32_bytes(1, 2)}})
    expect_raises(Error, /does not hold/) { SafeTensors.open(bad) }
  end

  it "rejects a header longer than the file" do
    bad = File.tempname("tensomite", ".safetensors")
    File.open(bad, "w") { |file| file.write_bytes(1000_u64, IO::ByteFormat::LittleEndian) }
    expect_raises(Error, /overruns/) { SafeTensors.open(bad) }
  end
end
