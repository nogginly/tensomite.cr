module Tensomite
  struct Tensor
    # Returns *tensors* joined along *axis*. Every other dimension must match.
    #
    # ```
    # Tensor.cat([a, b], 0) # [2, 3] and [1, 3] give [3, 3]
    # ```
    def self.cat(tensors : Array(Tensor), axis : Int32) : Tensor
      first = tensors.first? || raise ShapeError.new("cat of no tensors")
      first.check_compatible("cat", tensors)
      axis = first.view.normalize_axis(axis)
      tensors.each do |tensor|
        unless tensor.ndim == first.ndim && (0...tensor.ndim).all? { |i| i == axis || tensor.shape[i] == first.shape[i] }
          raise ShapeError.new("cannot cat #{first.shape} and #{tensor.shape} along axis #{axis}")
        end
      end
      shape = first.shape.dup
      shape[axis] = tensors.sum(&.shape[axis])
      Tensor.lazy(Node.new(Op::Cat, tensors, shape, first.dtype, first.device, axis))
    end

    # Returns the rows of this 2-D table selected by *ids*, an `I32` tensor of any shape.
    # The result has shape `ids.shape + [width]`.
    #
    # An id outside the table raises `ExecutionError` when the result is realised.
    def gather(ids : Tensor) : Tensor
      raise ShapeError.new("gather needs a 2-D table, not #{shape}") unless ndim == 2
      raise DTypeError.new("gather ids must be I32, not #{ids.dtype}") unless ids.dtype.i32?
      raise DeviceError.new("gather of tensors on different devices") unless device.same?(ids.device)
      Tensor.lazy(Node.new(Op::Gather, [self, ids], ids.shape + [shape[1]], dtype, device))
    end

    # Returns the matrix product over the last two dimensions, `[..., M, K]` by
    # `[..., K, N]` giving `[..., M, N]`. Leading batch dimensions broadcast.
    def matmul(other : Tensor) : Tensor
      check_compatible("matmul", {other})
      unless ndim >= 2 && other.ndim >= 2 && shape[-1] == other.shape[-2]
        raise ShapeError.new("cannot matmul #{shape} by #{other.shape}")
      end
      batch = View.broadcast_shape(shape[0...-2], other.shape[0...-2])
      a = expand(batch + shape[-2..])
      b = other.expand(batch + other.shape[-2..])
      Tensor.lazy(Node.new(Op::MatMul, [a, b], batch + [shape[-2], other.shape[-1]], dtype, device))
    end
  end
end
