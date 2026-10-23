require "json"

module Tensomite
  # Reads tensors from a [safetensors](https://github.com/huggingface/safetensors) file.
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
