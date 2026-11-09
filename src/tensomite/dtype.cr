module Tensomite
  # The element type of a tensor, known at runtime.
  #
  # `Q8_0`, `Q4_0` and `Q4_1` are ggml's block formats: each block of 32
  # consecutive elements of a row is stored as integers sharing an `F16` scale
  # (and, for `Q4_1`, an `F16` minimum). Tensors in them are leaves, read from
  # files or made by `Tensor#quantize` (Decision N).
  enum DType
    F32
    I32
    Q8_0
    Q4_0
    Q4_1

    # Returns the number of bytes that hold *count* elements.
    #
    # Every byte-size calculation goes through here, because block formats store a
    # group of elements in a fixed number of bytes rather than a size per element.
    #
    # Raises `ShapeError` for a block format and a count that is not a whole
    # number of blocks.
    def bytes_for(count : Int32) : Int32
      case self
      in F32, I32 then count * 4
      in Q8_0, Q4_0, Q4_1
        raise ShapeError.new("#{count} elements are not whole blocks of #{DType.block_size} for #{self}") unless count % DType.block_size == 0
        count // DType.block_size * block_bytes
      end
    end

    # Returns the number of elements in each block of a block format.
    def self.block_size : Int32
      32
    end

    # Returns the bytes one block takes in a block format: its scales, then
    # its 32 quantised elements.
    def block_bytes : Int32
      case self
      in Q8_0     then 2 + 32
      in Q4_0     then 2 + 16
      in Q4_1     then 4 + 16
      in F32, I32 then raise DTypeError.new("#{self} is not a block format")
      end
    end

    # Returns true for the block formats.
    def block? : Bool
      q8_0? || q4_0? || q4_1?
    end

    # Returns true for floating-point types. Block formats are not: no op
    # computes in them.
    def float? : Bool
      f32?
    end

    # Returns the `DType` matching a Crystal type, e.g. `DType.of(Float32) # => DType::F32`.
    def self.of(type : Float32.class) : DType
      F32
    end

    # :ditto:
    def self.of(type : Int32.class) : DType
      I32
    end
  end
end
