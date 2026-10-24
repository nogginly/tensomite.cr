require "json"

module Tensomite
  # Reads and writes [safetensors](https://github.com/huggingface/safetensors) files.
  #
  # The file is memory-mapped once. On a `HostDevice` each tensor whose data is
  # 4-byte aligned is a realised leaf over its slice of the mapping, so nothing is
  # copied or read until used. Other tensors are copied into a new buffer: older
  # writers do not pad the header, which leaves every tensor unaligned.
  #
  # ```
  # weights = SafeTensors.open("model.safetensors")
  # weights.names               # => ["wte.weight", ...]
  # weights["wte.weight"].shape # => [50257, 768]
  # ```
  class SafeTensors
    # The dtype, shape and byte range, within the data section, of one tensor.
    record Entry, dtype : DType, shape : Array(Int32), start : Int64, stop : Int64

    getter path : String
    @mapping : MappedFile
    @data_start : Int64

    # Returns the metadata of every tensor, by name.
    getter entries : Hash(String, Entry)

    # Opens the file at *path* and reads its header.
    #
    # Raises `Error` for a malformed header and `DTypeError` for a dtype tensomite
    # does not support yet.
    def self.open(path : String) : SafeTensors
      new(path)
    end

    private def initialize(@path : String)
      @mapping = MappedFile.new(path)
      header_size = IO::ByteFormat::LittleEndian.decode(UInt64, @mapping.bytes(0, 8)).to_i64
      unless 8 + header_size <= @mapping.size
        raise Error.new("#{path}: header of #{header_size} bytes overruns a #{@mapping.size}-byte file")
      end
      @data_start = 8_i64 + header_size
      @entries = parse_header(String.new(@mapping.bytes(8, header_size.to_i32)))
    end

    # Writes *tensors* to a safetensors file at *path*, in the order given, with
    # optional string *metadata* under the header's `__metadata__` key.
    #
    # Each tensor is written as its values in row-major order, so a view is saved
    # as the tensor it shows, in the host's byte order (little-endian on every
    # platform tensomite supports). The header is padded with spaces to a multiple
    # of 8 bytes, so that every tensor's data is aligned and can be mapped without
    # a copy when read back.
    #
    # The file is written beside *path* and then renamed over it. A file already at
    # *path* is replaced whole, never rewritten in place, so tensors still mapped
    # from it, such as a model loaded from the same path, keep their values.
    def self.write(path : String, tensors : Hash(String, Tensor), metadata : Hash(String, String)? = nil) : Nil
      offset = 0_i64
      header = JSON.build do |json|
        json.object do
          if metadata
            json.field("__metadata__") { metadata.to_json(json) }
          end
          tensors.each do |name, tensor|
            size = tensor.dtype.bytes_for(tensor.size).to_i64
            json.field(name) do
              json.object do
                json.field "dtype", tensor.dtype.to_s
                json.field "shape", tensor.shape
                json.field "data_offsets", [offset, offset + size]
              end
            end
            offset += size
          end
        end
      end
      header += " " * ((8 - header.bytesize % 8) % 8)
      temp = "#{path}.#{Process.pid}.#{Random::Secure.hex(4)}.tmp"
      begin
        File.open(temp, "w") do |file|
          file.write_bytes(header.bytesize.to_u64, IO::ByteFormat::LittleEndian)
          file << header
          tensors.each_value { |tensor| file.write(contents(tensor)) }
        end
        File.rename(temp, path)
      ensure
        File.delete?(temp)
      end
    end

    # Returns *tensor*'s values in row-major order, as bytes.
    private def self.contents(tensor : Tensor) : Bytes
      case tensor.dtype
      in .f32? then tensor.to_a(Float32).to_unsafe.to_slice(tensor.size).to_unsafe_bytes
      in .i32? then tensor.to_a(Int32).to_unsafe.to_slice(tensor.size).to_unsafe_bytes
      end
    end

    # Returns the names of all tensors, in file order.
    def names : Array(String)
      entries.keys
    end

    def has_key?(name : String) : Bool
      entries.has_key?(name)
    end

    # Returns tensor *name* as a realised leaf on *device*; raises `KeyError` if absent.
    def [](name : String, device : Device = CPU.default) : Tensor
      entry = entries[name]
      length = entry.stop - entry.start
      raise Error.new("#{name}: #{length} bytes is over the 2 GiB limit for one tensor") if length > Int32::MAX
      bytes = @mapping.bytes(@data_start + entry.start, length.to_i32)
      buffer =
        if device.is_a?(HostDevice) && bytes.to_unsafe.address % 4 == 0
          device.adopt(bytes, @mapping)
        else
          device.alloc(bytes.size).tap { |buf| device.upload(buf, bytes) }
        end
      Tensor.new(Node.new(buffer, entry.shape, entry.dtype, device), View.contiguous(entry.shape))
    end

    # Parses the JSON header and checks every entry against the file.
    #
    # 1. Maps each safetensors dtype name onto a `DType`.
    # 2. Checks that each byte range fits the data section, holds exactly the bytes
    #    its shape needs.
    private def parse_header(json : String) : Hash(String, Entry)
      data_size = @mapping.size - @data_start
      result = {} of String => Entry
      JSON.parse(json).as_h.each do |name, info|
        next if name == "__metadata__"
        dtype = case code = info["dtype"].as_s
                when "F32" then DType::F32
                when "I32" then DType::I32
                else            raise DTypeError.new("#{name}: safetensors dtype #{code} is not supported yet")
                end
        shape = info["shape"].as_a.map(&.as_i)
        start, stop = info["data_offsets"].as_a.map(&.as_i64)
        unless 0 <= start <= stop <= data_size && stop - start == dtype.bytes_for(shape.product(1)).to_i64
          raise Error.new("#{name}: byte range #{start}...#{stop} does not hold #{dtype} #{shape}")
        end
        result[name] = Entry.new(dtype, shape, start, stop)
      end
      result
    end
  end
end
