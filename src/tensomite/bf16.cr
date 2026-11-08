module Tensomite
  # A bfloat16 number: the top 16 bits of a `Float32`, with the same exponent
  # range and 8 bits of precision. Most small instruct models ship their weights
  # in it. tensomite computes in `Float32`, so it only converts, both ways.
  #
  # ```
  # BF16.new(1.0_f32).bits        # => 0x3f80
  # BF16.from_bits(0x3f80).to_f32 # => 1.0
  # ```
  struct BF16
    getter bits : UInt16

    # Returns the bfloat16 whose bits are *bits*.
    def self.from_bits(bits : UInt16) : BF16
      new(bits)
    end

    private def initialize(@bits : UInt16)
    end

    # Returns *value* rounded to the nearest bfloat16, ties to even, as PyTorch
    # rounds. A NaN stays a NaN.
    def self.new(value : Float32) : BF16
      word = value.unsafe_as(UInt32)
      return new(((word >> 16) | 0x40).to_u16) if value.nan?
      new(((word &+ 0x7fff_u32 &+ ((word >> 16) & 1)) >> 16).to_u16)
    end

    # Returns the value exactly, as a `Float32`.
    def to_f32 : Float32
      (bits.to_u32 << 16).unsafe_as(Float32)
    end

    # Widens bfloat16 values, as *bytes* in little-endian order, into *values*.
    def self.widen(bytes : Bytes, values : Slice(Float32)) : Nil
      raise ArgumentError.new("#{bytes.size} bytes do not hold #{values.size} bfloat16 values") unless bytes.size == 2 * values.size
      words = values.to_unsafe.as(UInt32*)
      values.size.times do |i|
        words[i] = (bytes[2 * i].to_u32 | bytes[2 * i + 1].to_u32 << 8) << 16
      end
    end
  end
end
