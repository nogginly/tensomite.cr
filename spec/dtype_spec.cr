require "./spec_helper"

describe DType do
  it "maps Crystal types" do
    DType.of(Float32).should eq DType::F32
    DType.of(Int32).should eq DType::I32
  end

  it "computes the bytes that hold a number of elements" do
    DType::F32.bytes_for(3).should eq 12
    DType::I32.bytes_for(0).should eq 0
  end

  it "knows which types are floats" do
    DType::F32.float?.should be_true
    DType::I32.float?.should be_false
  end
end
