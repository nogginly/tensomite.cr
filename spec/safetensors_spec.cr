require "./spec_helper"
require "json"

# Writes a safetensors file holding *tensors* (name to dtype, shape and raw bytes)
# and returns its path. With *pad* false the header is padded so that the data
# section starts 3 bytes past a 4-byte boundary, as some older writers leave it.
private def write_safetensors(tensors : Hash(String, {String, Array(Int32), Bytes}), pad : Bool = true) : String
  offset = 0
  meta = tensors.transform_values do |(dtype, shape, bytes)|
    entry = {"dtype" => JSON::Any.new(dtype), "shape" => JSON::Any.new(shape.map { |dim| JSON::Any.new(dim.to_i64) }),
             "data_offsets" => JSON::Any.new([JSON::Any.new(offset.to_i64), JSON::Any.new((offset + bytes.size).to_i64)])}
    offset += bytes.size
    JSON::Any.new(entry)
  end
  json = meta.to_json
  json += " " * (pad ? (8 - json.bytesize % 8) % 8 : (3 - json.bytesize % 4) % 4)
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

  it "copies tensors whose data is not 4-byte aligned" do
    unaligned = SafeTensors.open(write_safetensors({"a" => {"F32", [2, 2], f32_bytes(1, 2, 3, 4)}}, pad: false))
    a = unaligned["a"]
    (CPU.default.map(a.node.buffer!).not_nil!("CPU memory is host-visible").to_unsafe.address % 4).should eq 0
    a.to_a(Float32).should eq f32(1, 2, 3, 4)
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

describe "SafeTensors.write" do
  it "writes tensors that read back with the same names, dtypes, shapes and values" do
    path = File.tempname("tensomite", ".safetensors")
    a = Tensor.from(f32(1, 2, 3, 4, 5, 6), [2, 3])
    SafeTensors.write(path, {"a" => a, "ids" => Tensor.from([7, 8, 9], [3]), "scalar" => Tensor.from(f32(2.5), [] of Int32)})
    file = SafeTensors.open(path)
    file.names.should eq ["a", "ids", "scalar"]
    file.entries["ids"].dtype.should eq DType::I32
    file["a"].shape.should eq [2, 3]
    file["a"].to_a(Float32).should eq f32(1, 2, 3, 4, 5, 6)
    file["ids"].to_a(Int32).should eq [7, 8, 9]
    file["scalar"].to_a(Float32).should eq f32(2.5)
  end

  it "writes a view as the tensor it shows" do
    path = File.tempname("tensomite", ".safetensors")
    SafeTensors.write(path, {"t" => Tensor.from(f32(1, 2, 3, 4, 5, 6), [2, 3]).transpose})
    file = SafeTensors.open(path)
    file["t"].shape.should eq [3, 2]
    file["t"].to_a(Float32).should eq f32(1, 4, 2, 5, 3, 6)
  end

  it "pads the header so that every tensor is aligned, and keeps metadata" do
    path = File.tempname("tensomite", ".safetensors")
    SafeTensors.write(path, {"a" => Tensor.from(f32(1), [1]), "b" => Tensor.from(f32(2, 3), [2])}, {"format" => "pt"})
    bytes = File.read(path).to_slice
    header_size = IO::ByteFormat::LittleEndian.decode(UInt64, bytes[0, 8])
    (header_size % 8).should eq 0
    header = JSON.parse(String.new(bytes[8, header_size.to_i]))
    header["__metadata__"].as_h.should eq({"format" => JSON::Any.new("pt")})
    SafeTensors.open(path)["b"].to_a(Float32).should eq f32(2, 3)
  end

  it "replaces a file whose tensors are still mapped without changing them" do
    path = File.tempname("tensomite", ".safetensors")
    SafeTensors.write(path, {"a" => Tensor.from(f32(1, 2), [2])})
    mapped = SafeTensors.open(path)["a"]
    SafeTensors.write(path, {"a" => Tensor.from(f32(8, 9), [2])})
    mapped.to_a(Float32).should eq f32(1, 2)
    SafeTensors.open(path)["a"].to_a(Float32).should eq f32(8, 9)
    Dir.glob("#{path}.*.tmp").should be_empty
  end
end
