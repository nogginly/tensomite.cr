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
      Tensor.from_blocks(Quant.quantize_memory(dtype, Slice.new(values.to_unsafe, values.size)), shape, dtype, device)
    end

    # Returns this block-format tensor's values as a new realised `F32` leaf on
    # the same device.
    def dequantize : Tensor
      raise DTypeError.new("#{dtype} is not a block format") unless dtype.block?
      values = Array(Float32).new(size, 0.0_f32)
      Quant.dequantize(dtype, block_bytes, Slice.new(values.to_unsafe, values.size))
      Tensor.from(values, shape, device)
    end

    # Returns a new realised leaf of this block-format tensor's rows in *order*:
    # row `i` of the result is row `order[i]` of this one.
    def take_rows(order : Array(Int32)) : Tensor
      raise DTypeError.new("take_rows is for block formats, not #{dtype}") unless dtype.block?
      rows, width = shape[0], shape[1..].product(1)
      row_bytes = dtype.bytes_for(width)
      source = block_bytes
      bytes = Memory.allocate(row_bytes * order.size)
      order.each_with_index do |row, i|
        raise IndexError.new("row #{row} of #{rows}") unless 0 <= row < rows
        source[row_bytes * row, row_bytes].copy_to(bytes[row_bytes * i, row_bytes])
      end
      Tensor.from_blocks(bytes, [order.size] + shape[1..], dtype, device)
    end

    # Returns a new realised leaf of *tensors*' rows, in order: block-format
    # tensors of one dtype and row width.
    def self.cat_rows(tensors : Array(Tensor)) : Tensor
      first = tensors.first
      unless tensors.all? { |tensor| tensor.dtype == first.dtype && tensor.ndim == 2 && tensor.shape[1] == first.shape[1] }
        raise ShapeError.new("cat_rows needs 2-D tensors of one dtype and width")
      end
      parts = tensors.map(&.block_bytes)
      bytes = Memory.allocate(parts.sum(&.size))
      start = 0_i64
      parts.each do |part|
        part.copy_to(bytes[start, part.size])
        start += part.size
      end
      from_blocks(bytes, [tensors.sum(&.shape[0]), first.shape[1]], first.dtype, first.device)
    end

    # Returns a realised leaf of *dtype*, a block format, holding *bytes*, its
    # blocks row by row.
    def self.from_blocks(bytes : Bytes | Memory, shape : Array(Int32), dtype : DType, device : Device = CPU.default) : Tensor
      bytes = Memory.new(bytes) if bytes.is_a?(Bytes)
      unless bytes.size == dtype.bytes_for(shape.product(1))
        raise ShapeError.new("#{bytes.size} bytes do not hold #{dtype} #{shape}")
      end
      buffer = device.alloc(bytes.size)
      device.upload(buffer, bytes)
      new(Node.new(buffer, shape, dtype, device), View.contiguous(shape))
    end

    # Returns this tensor's data in row-major order as a copy of its bytes:
    # each element's for `F32` and `I32`, and each block's for a block format,
    # as `Tensor.from_blocks` takes them. Raises `ArgumentError` for 2 GiB or
    # more; `#memory` has no such limit.
    def bytes : Bytes
      memory.bytes
    end

    # Returns a copy of this tensor's data in row-major order, as `#bytes` does,
    # in memory that may exceed 2 GiB.
    def memory : Memory
      return block_bytes if dtype.block?
      dense = contiguous.realize
      source = dense.host_memory[dtype.bytes_for(dense.view.offset), dtype.bytes_for(size)]
      Memory.allocate(source.size).tap { |copy| source.copy_to(copy) }
    end

    # Returns a copy of the bytes of the rows this block-format tensor views.
    protected def block_bytes : Memory
      source = realize.host_memory[dtype.bytes_for(view.offset), dtype.bytes_for(size)]
      Memory.allocate(source.size).tap { |copy| source.copy_to(copy) }
    end
  end
end
