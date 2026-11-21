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

    # Returns *tensors*, all of one shape, stacked along a new axis *axis*:
    # each gains a dimension of 1 there, and they are joined along it.
    #
    # ```
    # Tensor.stack([a, b], 0) # two [2, 3] give [2, 2, 3]
    # ```
    def self.stack(tensors : Array(Tensor), axis : Int32 = 0) : Tensor
      first = tensors.first? || raise ShapeError.new("stack of no tensors")
      unless tensors.all? { |tensor| tensor.shape == first.shape }
        raise ShapeError.new("cannot stack #{tensors.map(&.shape).uniq!.join(", ")}: shapes differ")
      end
      axis += first.ndim + 1 if axis < 0
      raise ShapeError.new("stack axis #{axis} out of range for #{first.ndim + 1} dimensions") unless 0 <= axis <= first.ndim
      shape = first.shape.dup.insert(axis, 1)
      cat(tensors.map(&.reshape(shape)), axis)
    end

    # Returns this tensor's slices along *axis*, each without that axis: the
    # inverse of `.stack`. Each is a view; no data is copied.
    #
    # ```
    # Tensor.stack([a, b], 0).unbind(0) # [a, b]
    # ```
    def unbind(axis : Int32 = 0) : Array(Tensor)
      axis = view.normalize_axis(axis)
      shape = self.shape.dup.tap(&.delete_at(axis))
      Array.new(self.shape[axis]) { |i| slice(axis, i, i + 1).reshape(shape) }
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

    # Returns this tensor, `[rows, heads, capacity, width]`, with each token of
    # *value*, `[tokens, heads, width]`, written in place at its *slots* entry,
    # `row * capacity + position`: the in-place write a cache of many rows makes
    # for one step's new keys or values (see `Op::WriteRows`). As with
    # `#write`, this tensor must cover its buffer, and tensors read from it
    # before must be realised first.
    def write_rows(value : Tensor, slots : Array(Int32)) : Tensor
      check_compatible("write_rows", {value})
      unless ndim == 4 && value.ndim == 3 && value.shape[0] == slots.size && value.shape[1] == shape[1] && value.shape[2] == shape[3]
        raise ShapeError.new("cannot write #{value.shape} into rows of #{shape} at #{slots.size} slots")
      end
      raise ShapeError.new("write_rows needs a tensor that covers its buffer") unless view.contiguous? && view.offset == 0 && size == node.size
      places = shape[0] * shape[2]
      slots.each { |slot| raise ShapeError.new("slot #{slot} outside #{shape[0]} rows of #{shape[2]}") unless 0 <= slot < places }
      ids = Tensor.from(slots, [slots.size], device).named("slots")
      Tensor.lazy(Node.new(Op::WriteRows, [self, value, ids], shape, dtype, device))
    end

    # Raises unless *weight* and *bias* can apply to these rows in `#linear`.
    private def check_linear(weight : Tensor, bias : Tensor?) : Nil
      (bias ? {weight, bias} : {weight}).each do |other|
        raise DeviceError.new("linear of tensors on different devices") unless device.same?(other.device)
      end
      raise DTypeError.new("linear needs F32 rows, not #{dtype}") unless dtype.f32?
      unless ndim >= 1 && weight.ndim == 2 && shape[-1] == weight.shape[1]
        raise ShapeError.new("cannot apply a #{weight.shape} weight to #{shape}")
      end
      n = weight.shape[0]
      if bias && !(bias.shape == [n] && bias.dtype.f32?)
        raise ShapeError.new("a bias for #{n} outputs must be F32 [#{n}], not #{bias.dtype} #{bias.shape}")
      end
    end

    # Returns the rows of this 2-D table selected by *ids*, an `I32` tensor of any shape.
    # The result has shape `ids.shape + [width]`. A block-format table gives
    # `F32` rows, each dequantised.
    #
    # An id outside the table raises `ExecutionError` when the result is realised.
    def gather(ids : Tensor) : Tensor
      raise ShapeError.new("gather needs a 2-D table, not #{shape}") unless ndim == 2
      raise DTypeError.new("gather ids must be I32, not #{ids.dtype}") unless ids.dtype.i32?
      raise DeviceError.new("gather of tensors on different devices") unless device.same?(ids.device)
      Tensor.lazy(Node.new(Op::Gather, [self, ids], ids.shape + [shape[1]], dtype.block? ? DType::F32 : dtype, device))
    end

    # Returns one element of each row of this 2-D table, `[rows]`: the column
    # *ids*, `I32` `[rows]`, selects, or 0 for a row whose id is *ignore*. It
    # picks each row's target on the device, so the ids are never read on the
    # host.
    #
    # An id outside the columns that is not *ignore* raises `ExecutionError`
    # when the result is realised.
    #
    # ```
    # log_probs.pick(targets) # [rows]: each row's log-probability of its target
    # ```
    def pick(ids : Tensor, ignore : Int32? = nil) : Tensor
      check_pick("pick", ids, shape[0]? || 0)
      raise ShapeError.new("pick needs a 2-D table, not #{shape}") unless ndim == 2
      Tensor.lazy(Node.new(Op::Pick, [self, ids], [shape[0]], dtype, device, ignore: ignore))
    end

    # Returns the adjoint of `#pick`: a `[rows, columns]` tensor, zero but for
    # each row's element at the column *ids* selects, which holds this `[rows]`
    # tensor's element; a row whose id is *ignore* is all zero.
    def place(ids : Tensor, columns : Int32, ignore : Int32? = nil) : Tensor
      raise ShapeError.new("place needs [rows] values, not #{shape}") unless ndim == 1
      check_pick("place", ids, shape[0])
      Tensor.lazy(Node.new(Op::Place, [self, ids], [shape[0], columns], dtype, device, ignore: ignore))
    end

    private def check_pick(name : String, ids : Tensor, rows : Int32) : Nil
      raise DTypeError.new("#{name} ids must be I32, not #{ids.dtype}") unless ids.dtype.i32?
      raise DeviceError.new("#{name} of tensors on different devices") unless device.same?(ids.device)
      raise ShapeError.new("#{name} needs one id per row, [#{rows}], not #{ids.shape}") unless ids.shape == [rows]
    end

    # Returns `x · weightᵀ (+ bias)` for rows `x` of `[..., K]` and *weight* of
    # `[N, K]`, as PyTorch's `linear` does, so a weight stored as GGUF and
    # PyTorch store it, `[outputs, inputs]`, needs no transpose.
    #
    # With a block-format weight it is one `BlockMatMul`, which quantises each
    # row of `x` to 8 bits; with an `F32` weight it is
    # `matmul(weight.transpose, bias)`.
    #
    # ```
    # w = Tensor.from(values, [1536, 576]).quantize(DType::Q4_0)
    # x.linear(w) # [rows, 1536]
    # ```
    def linear(weight : Tensor, bias : Tensor? = nil) : Tensor
      return matmul(weight.transpose, bias) unless weight.dtype.block?
      check_linear(weight, bias)
      n = weight.shape[0]
      srcs = [self, weight]
      srcs << (bias.view.contiguous? ? bias : bias.contiguous) if bias
      Tensor.lazy(Node.new(Op::BlockMatMul, srcs, shape[0...-1] + [n], DType::F32, device))
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
