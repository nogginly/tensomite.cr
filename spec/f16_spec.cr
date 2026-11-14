require "./spec_helper"

describe F16 do
  it "round-trips every half that is not a NaN" do
    (0..0xffff).each do |bits|
      half = F16.from_bits(bits.to_u16)
      next if half.to_f32.nan?
      F16.new(half.to_f32).bits.should eq bits
    end
  end

  it "widens every half to the value its bits define" do
    (0..0xffff).each do |bits|
      sign = bits >> 15 == 1 ? -1.0 : 1.0
      exponent, fraction = (bits >> 10) & 0x1f, bits & 0x3ff
      got = F16.from_bits(bits.to_u16).to_f32
      case exponent
      when  0 then got.to_f64.should eq sign * fraction * 2.0 ** -24
      when 31 then fraction == 0 ? got.should(eq(sign * Float32::INFINITY)) : got.unsafe_as(UInt32).should(eq((bits.to_u32 & 0x8000) << 16 | 0x7f80_0000 | fraction.to_u32 << 13))
      else         got.to_f64.should eq sign * (1 + fraction / 1024) * 2.0 ** (exponent - 15)
      end
    end
  end

  it "converts known values" do
    F16.new(1.0_f32).bits.should eq 0x3c00
    F16.new(-2.0_f32).bits.should eq 0xc000
    F16.new(65504.0_f32).bits.should eq 0x7bff
    F16.from_bits(0x0001).to_f32.should eq 5.9604645e-8_f32
    F16.from_bits(0x03ff).to_f32.should eq 6.097555e-5_f32
  end

  it "rounds to nearest, ties to even, and overflows to infinity" do
    F16.new(1.00048828125_f32).bits.should eq 0x3c00
    F16.new(1.00146484375_f32).bits.should eq 0x3c02
    F16.new(65519.0_f32).bits.should eq 0x7bff
    F16.new(65520.0_f32).bits.should eq 0x7c00
    F16.new(2.9802322e-8_f32).bits.should eq 0x0000
    F16.new(8.940697e-8_f32).bits.should eq 0x0002
    F16.new(Float32::NAN).to_f32.nan?.should be_true
    F16.new(-Float32::INFINITY).bits.should eq 0xfc00
  end

  it "widens little-endian bytes" do
    values = Slice(Float32).new(2)
    F16.widen(Bytes[0x00, 0x3c, 0x00, 0xc1], values)
    values.to_a.should eq [1.0_f32, -2.5_f32]
  end
end
