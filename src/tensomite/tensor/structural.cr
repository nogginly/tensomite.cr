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

    # Returns this tensor with *value* written in place at positions *start*
    # onwards along *axis*, as a tensor sharing its buffer. *value* matches this
    # tensor's shape except along *axis*.
    #
    # The only op that changes a buffer, for caches that grow without copying,
    # such as `CausalSelfAttention::Cache`. This tensor must cover its whole
    # buffer, and any tensor read from it before the write must be realised
    # first, or it may see the new values.
    #
    # ```
    # cache = Tensor.zeros([4, 2])
    # cache = cache.write(Tensor.from([1f32, 2f32], [1, 2]), 0, 1) # row 1 is now [1, 2]
    # ```
    def write(value : Tensor, axis : Int32, start : Int32) : Tensor
      check_compatible("write", {value})
      axis = view.normalize_axis(axis)
      stop = start + value.shape[axis]
      unless value.ndim == ndim && (0...ndim).all? { |i| i == axis || value.shape[i] == shape[i] } && 0 <= start && stop <= shape[axis]
        raise ShapeError.new("cannot write #{value.shape} into #{shape} at #{start} along axis #{axis}")
      end
      raise ShapeError.new("write needs a tensor that covers its buffer") unless view.contiguous? && view.offset == 0 && size == node.size
      region = View.contiguous(shape).slice(axis, start, stop)
      Tensor.lazy(Node.new(Op::Write, [self, value], shape, dtype, device, source_view: region))
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
    #
    # With a *bias*, returns the product plus *bias*, with the same bits as
    # `a.matmul(b) + bias`. A contiguous `[N]` bias, as a `Linear` layer has, is
    # added inside the product, saving an op; any other is added after it.
    def matmul(other : Tensor, bias : Tensor? = nil) : Tensor
      check_compatible("matmul", bias ? {other, bias} : {other})
      unless ndim >= 2 && other.ndim >= 2 && shape[-1] == other.shape[-2]
        raise ShapeError.new("cannot matmul #{shape} by #{other.shape}")
      end
      batch = View.broadcast_shape(shape[0...-2], other.shape[0...-2])
      a = expand(batch + shape[-2..])
      b = other.expand(batch + other.shape[-2..])
      result = batch + [shape[-2], other.shape[-1]]
      if bias && !(bias.shape == [other.shape[-1]] && bias.view.contiguous?)
        return Tensor.lazy(Node.new(Op::MatMul, [a, b], result, dtype, device)) + bias.expand(result)
      end
      srcs = bias ? [a, b, bias] : [a, b]
      Tensor.lazy(Node.new(Op::MatMul, srcs, result, dtype, device))
    end
  end
end
