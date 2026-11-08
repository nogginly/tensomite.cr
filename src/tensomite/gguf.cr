require "json"

module Tensomite
  # Reads [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
  # files, llama.cpp's format, which holds a model's weights, its
  # configuration and its tokeniser in one file.
  #
  # The file is memory-mapped once. Its metadata is read whole, each value as
  # `JSON::Any`. Tensors are read when asked for: `F32` tensors and block
  # formats (`Q8_0`, `Q4_0`, `Q4_1`) on a `HostDevice` as realised leaves over
  # the mapping, and `F16` and `BF16` tensors widened to `F32` copies.
  #
  # ```
  # file = GGUF.open("model.gguf")
  # file.metadata["llama.block_count"].as_i # => 30
  # file.entries["token_embd.weight"].shape # => [49152, 576]
  # file["output_norm.weight"]              # an F32 tensor
  # ```
  #
  # Shapes are in tensomite's order, outermost first: GGUF lists a tensor's
  # dimensions innermost first, so they are reversed.
  class GGUF
    MAGIC = 0x46554747_u32

    # The kinds of tensor data, by ggml's numbering.
    enum Kind : UInt32
      F32  =  0
      F16  =  1
      Q4_0 =  2
      Q4_1 =  3
      Q8_0 =  8
      BF16 = 30

      # Returns the bytes that hold *count* elements.
      def bytes_for(count : Int64) : Int64
        case self
        in F32       then count * 4
        in F16, BF16 then count * 2
        in Q4_0      then count // 32 * 18
        in Q4_1      then count // 32 * 20
        in Q8_0      then count // 32 * 34
        end
      end

      # Returns the number of elements in one block of the format.
      def block : Int32
        quantised? ? 32 : 1
      end

      def quantised? : Bool
        q4_0? || q4_1? || q8_0?
      end

      # Returns the `DType` a tensor of this kind is read as.
      def dtype : DType
        case self
        in Q8_0           then DType::Q8_0
        in Q4_0           then DType::Q4_0
        in Q4_1           then DType::Q4_1
        in F32, F16, BF16 then DType::F32
        end
      end
    end

    # The kind, shape and absolute byte offset of one tensor.
    record Entry, kind : Kind, shape : Array(Int32), offset : Int64 do
      # Returns the number of bytes its data takes.
      def bytesize : Int64
        kind.bytes_for(shape.product(1_i64))
      end
    end

    getter path : String

    # Returns every metadata value, by key, as `general.architecture`.
    getter metadata : Hash(String, JSON::Any)

    # Returns every tensor's kind, shape and offset, by name, in file order.
    getter entries : Hash(String, Entry)

    @mapping : MappedFile

    # Opens the file at *path* and reads its header, metadata and tensor list.
    #
    # Raises `Error` for a file that is not GGUF version 2 or 3 or is
    # malformed, and `DTypeError` for a tensor kind tensomite cannot read.
    def self.open(path : String) : GGUF
      new(path)
    end

    private def initialize(@path : String)
      @mapping = MappedFile.new(path)
      @metadata = {} of String => JSON::Any
      @entries = {} of String => Entry
      parse
    end

    # Returns the names of all tensors, in file order.
    def names : Array(String)
      entries.keys
    end

    def has_key?(name : String) : Bool
      entries.has_key?(name)
    end

    # Returns tensor *name* as a realised leaf on *device*, of `F32` or its
    # block format; raises `KeyError` if absent.
    def [](name : String, device : Device = CPU.default) : Tensor
      entry = entries[name]
      if entry.kind.quantised?
        raise Error.new("#{name}: #{entry.bytesize} bytes is over the 2 GiB limit for one tensor") if entry.bytesize > Int32::MAX
        bytes = @mapping.bytes(entry.offset, entry.bytesize.to_i32)
        buffer = device.is_a?(HostDevice) ? device.adopt(bytes, @mapping) : device.alloc(bytes.size).tap { |buf| device.upload(buf, bytes) }
        return Tensor.new(Node.new(buffer, entry.shape, entry.kind.dtype, device), View.contiguous(entry.shape))
      end
      count = entry.shape.product(1_i64)
      raise Error.new("#{name}: #{count * 4} bytes is over the 2 GiB limit for one tensor") if count * 4 > Int32::MAX
      bytes = @mapping.bytes(entry.offset, entry.bytesize.to_i32)
      buffer =
        if entry.kind.f32?
          if device.is_a?(HostDevice) && bytes.to_unsafe.address % 4 == 0
            device.adopt(bytes, @mapping)
          else
            device.alloc(bytes.size).tap { |buf| device.upload(buf, bytes) }
          end
        else
          values = Slice(Float32).new(count.to_i32)
          entry.kind.f16? ? F16.widen(bytes, values) : BF16.widen(bytes, values)
          device.alloc(values.bytesize).tap { |buf| device.upload(buf, values.to_unsafe_bytes) }
        end
      Tensor.new(Node.new(buffer, entry.shape, DType::F32, device), View.contiguous(entry.shape))
    end

    # Returns the raw bytes of tensor *name*, without copying.
    def bytes(name : String) : Bytes
      entry = entries[name]
      @mapping.bytes(entry.offset, entry.bytesize.to_i32)
    end

    # Reads the file in order:
    #
    # 1. The magic number, the version (2 or 3), and the tensor and metadata
    #    counts.
    # 2. Each metadata key and its typed value.
    # 3. Each tensor's name, dimensions, kind and offset within the data.
    # 4. The data section, which starts at the next multiple of
    #    `general.alignment` (32 by default); every offset must lie in it.
    private def parse : Nil
      reader = Reader.new(@mapping)
      raise Error.new("#{path} is not a GGUF file") unless reader.u32 == MAGIC
      version = reader.u32
      raise Error.new("#{path}: GGUF version #{version} is not supported") unless version.in?(2, 3)
      tensors, keys = reader.u64, reader.u64
      keys.times do
        key = reader.string
        @metadata[key] = reader.value(reader.u32)
      end
      infos = Array(Tuple(String, Array(Int32), UInt32, Int64)).new(tensors.to_i) do
        name = reader.string
        dims = Array.new(reader.u32) { reader.u64.to_i32 }
        {name, dims.reverse, reader.u32, reader.u64.to_i64}
      end
      alignment = @metadata["general.alignment"]?.try(&.as_i64) || 32_i64
      start = (reader.position + alignment - 1) // alignment * alignment
      infos.each do |name, shape, code, offset|
        kind = Kind.from_value?(code) || raise DTypeError.new("#{name}: GGUF tensor type #{code} is not supported yet")
        if kind.quantised? && shape.last % kind.block != 0
          raise Error.new("#{name}: rows of #{shape.last} do not split into blocks of #{kind.block}")
        end
        entry = Entry.new(kind, shape, start + offset)
        unless offset % alignment == 0 && entry.offset + entry.bytesize <= @mapping.size
          raise Error.new("#{name}: #{entry.bytesize} bytes at #{offset} do not fit #{path}")
        end
        @entries[name] = entry
      end
    end

    # Reads little-endian values from the start of a mapping, in turn.
    private class Reader
      getter position = 0_i64

      def initialize(@mapping : MappedFile)
      end

      def u32 : UInt32
        IO::ByteFormat::LittleEndian.decode(UInt32, take(4))
      end

      def u64 : UInt64
        IO::ByteFormat::LittleEndian.decode(UInt64, take(8))
      end

      def string : String
        String.new(take(u64.to_i32))
      end

      # Returns a value of GGUF value type *type*.
      def value(type : UInt32) : JSON::Any
        case type
        when 0..5, 10, 11 then JSON::Any.new(integer(type))
        when 6            then JSON::Any.new(IO::ByteFormat::LittleEndian.decode(Float32, take(4)).to_f64)
        when 7            then JSON::Any.new(take(1)[0] != 0)
        when 8            then JSON::Any.new(string)
        when 9
          element = u32
          JSON::Any.new(Array.new(u64.to_i) { value(element) })
        when 12 then JSON::Any.new(IO::ByteFormat::LittleEndian.decode(Float64, take(8)))
        else         raise Error.new("GGUF value type #{type} is not known")
        end
      end

      # Returns an integer of GGUF value type *type*, one of the eight integer types.
      private def integer(type : UInt32) : Int64
        case type
        when  0 then take(1)[0].to_i64
        when  1 then take(1)[0].to_i8!.to_i64
        when  2 then IO::ByteFormat::LittleEndian.decode(UInt16, take(2)).to_i64
        when  3 then IO::ByteFormat::LittleEndian.decode(Int16, take(2)).to_i64
        when  4 then u32.to_i64
        when  5 then IO::ByteFormat::LittleEndian.decode(Int32, take(4)).to_i64
        when 10 then u64.to_i64!
        else         IO::ByteFormat::LittleEndian.decode(Int64, take(8))
        end
      end

      private def take(count : Int32) : Bytes
        bytes = @mapping.bytes(@position, count)
        @position += count
        bytes
      end
    end
  end
end
