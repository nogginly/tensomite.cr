module Tensomite
  struct Tensor
    # Returns this `F32` tensor quantised to *dtype*, a block format, as a new
    # realised leaf on the same device. Its last axis must be a whole number of
    # blocks of 32, and each block is quantised as ggml's reference code does.
    #
    # ```
    # w = Tensor.from(values, [576, 1536]).quantize(DType::Q4_0)
    # w.dequantize # within about a seventh of each block's largest magnitude
    # ```
    def quantize(dtype : DType) : Tensor
      raise DTypeError.new("#{dtype} is not a block format") unless dtype.block?
      raise DTypeError.new("quantize needs F32, not #{self.dtype}") unless self.dtype.f32?
      unless ndim > 0 && shape.last % DType.block_size == 0
        raise ShapeError.new("quantize needs rows of whole blocks of #{DType.block_size}, not #{shape}")
      end
      values = to_a(Float32)
      Tensor.from_blocks(Quant.quantize(dtype, Slice.new(values.to_unsafe, values.size)), shape, dtype, device)
    end

    # Returns this block-format tensor's values as a new realised `F32` leaf on
    # the same device.
    def dequantize : Tensor
      raise DTypeError.new("#{dtype} is not a block format") unless dtype.block?
      values = Array(Float32).new(size, 0.0_f32)
      Quant.dequantize(dtype, block_bytes, Slice.new(values.to_unsafe, values.size))
      Tensor.from(values, shape, device)
    end

    # Returns a realised leaf of *dtype*, a block format, holding *bytes*, its
    # blocks row by row.
    def self.from_blocks(bytes : Bytes, shape : Array(Int32), dtype : DType, device : Device = CPU.default) : Tensor
      unless bytes.size == dtype.bytes_for(shape.product(1))
        raise ShapeError.new("#{bytes.size} bytes do not hold #{dtype} #{shape}")
      end
      buffer = device.alloc(bytes.size)
      device.upload(buffer, bytes)
      new(Node.new(buffer, shape, dtype, device), View.contiguous(shape))
    end

    # Returns a copy of the bytes of the rows this block-format tensor views.
    protected def block_bytes : Bytes
      bytes = realize.host_bytes
      start = dtype.bytes_for(view.offset)
      bytes[start, dtype.bytes_for(size)].dup
    end
  end
end
