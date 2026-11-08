require "./spec_helper"

describe BF16 do
  it "keeps values that fit in 8 bits of precision" do
    [1.0_f32, -2.5_f32, 3.140625_f32, 0.0_f32, Float32::INFINITY, -Float32::INFINITY].each do |value|
      BF16.new(value).to_f32.should eq value
    end
    BF16.new(1.0_f32).bits.should eq 0x3f80
  end

  it "rounds to nearest, ties to even" do
    BF16.new(1.00390625_f32).to_f32.should eq 1.0_f32
    BF16.new(1.01171875_f32).to_f32.should eq 1.015625_f32
    BF16.new(1.0040_f32).to_f32.should eq 1.0078125_f32
  end

  it "keeps NaN a NaN" do
    BF16.new(Float32::NAN).to_f32.nan?.should be_true
  end

  it "widens little-endian bytes" do
    values = Slice(Float32).new(2)
    BF16.widen(Bytes[0x80, 0x3f, 0x20, 0xc0], values)
    values.to_a.should eq [1.0_f32, -2.5_f32]
  end
end
