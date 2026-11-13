module Tensomite
  # An IEEE 754 half-precision number: 5 bits of exponent and 10 of fraction,
  # as GGUF files store weights and quantised blocks store their scales.
  # tensomite computes in `Float32`, so it only converts, both ways.
  #
  # ```
  # F16.new(1.0_f32).bits        # => 0x3c00
  # F16.from_bits(0x3c00).to_f32 # => 1.0
  # ```
  struct F16
    getter bits : UInt16

    # Returns the half whose bits are *bits*.
    def self.from_bits(bits : UInt16) : F16
      new(bits)
    end

    private def initialize(@bits : UInt16)
    end

    # Returns *value* rounded to the nearest half, ties to even. Values beyond
    # ±65504 by half a step or more become infinite, and a NaN stays a NaN.
    #
    # 1. Infinities and NaNs keep their kind; values of 65520 or more overflow.
    # 2. Values below 2⁻¹⁴ become subnormal: the value times 2²⁴, rounded to
    #    an integer, is the fraction.
    # 3. Others drop 13 fraction bits, rounding to even, and rebias the
    #    exponent from 127 to 15; a carry out of the fraction raises it.
    def self.new(value : Float32) : F16
      word = value.unsafe_as(UInt32)
      sign = ((word >> 16) & 0x8000).to_u16
      magnitude = word & 0x7fff_ffff
      if magnitude >= 0x7f80_0000
        return new(sign | (magnitude > 0x7f80_0000 ? 0x7e00_u16 : 0x7c00_u16))
      end
      return new(sign | 0x7c00_u16) if magnitude >= 0x477f_f000
      if magnitude < 0x3880_0000
        return new(sign | (magnitude.unsafe_as(Float32) * 16_777_216.0_f32).round(:ties_even).to_u16)
      end
      new(sign | ((magnitude - 0x3800_0000 + 0xfff + ((magnitude >> 13) & 1)) >> 13).to_u16)
    end

    # Returns the value exactly, as a `Float32`, without branching:
    #
    # 1. Moves the exponent and fraction bits to a `Float32`'s positions.
    # 2. Multiplies by 2¹¹², which rebiases the exponent from 15 to 127 and
    #    normalises a subnormal half; a power of two multiplies exactly.
    # 3. Gives infinities and NaNs an exponent of all ones, keeping a NaN's
    #    fraction, and restores the sign.
    def to_f32 : Float32
      magnitude = bits.to_u32 & 0x7fff
      value = (magnitude << 13).unsafe_as(Float32) * 5.192297e+33_f32
      word = value.unsafe_as(UInt32) | (bits.to_u32 & 0x8000) << 16
      word |= 0x7f80_0000 if magnitude >= 0x7c00
      word.unsafe_as(Float32)
    end

    # Widens halves, as *bytes* in little-endian order, into *values*.
    def self.widen(bytes : Bytes, values : Slice(Float32)) : Nil
      raise ArgumentError.new("#{bytes.size} bytes do not hold #{values.size} halves") unless bytes.size == 2 * values.size
      values.size.times do |i|
        values[i] = from_bits(bytes[2 * i].to_u16 | bytes[2 * i + 1].to_u16 << 8).to_f32
      end
    end
  end
end
