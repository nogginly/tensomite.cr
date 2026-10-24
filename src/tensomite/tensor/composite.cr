module Tensomite
  struct Tensor
    # Returns the softmax along *axis*: `exp(x - max) / sum(exp(x - max))`.
    # Subtracting the maximum keeps `exp` from overflowing without changing the result.
    #
    # Along the last axis it is one fused op (`Op::Softmax`), which computes what
    # the composite computes, operation for operation, on host devices.
    def softmax(axis : Int32 = -1) : Tensor
      return row_op(Op::Softmax) if fusable_row?(axis)
      e = (self - max(axis, keepdims: true)).exp
      e / e.sum(axis, keepdims: true)
    end

    # Returns the log of the softmax along *axis*: `x - max - log(sum(exp(x - max)))`.
    # Fused along the last axis, as `#softmax` is.
    def log_softmax(axis : Int32 = -1) : Tensor
      return row_op(Op::LogSoftmax) if fusable_row?(axis)
      shifted = self - max(axis, keepdims: true)
      shifted - shifted.exp.sum(axis, keepdims: true).log
    end

    # Returns the mean cross-entropy of these `[n, classes]` logits against *targets*,
    # one class index per row: the mean over rows of `-log_softmax[row, target]`.
    #
    # The targets are read on the host to pick each row's entry, so *targets* is
    # realised when this is called.
    def cross_entropy(targets : Tensor) : Tensor
      raise ShapeError.new("cross_entropy needs [n, classes] logits, not #{shape}") unless ndim == 2
      rows, classes = shape
      raise ShapeError.new("#{targets.size} targets for #{rows} rows") unless targets.size == rows
      raise DTypeError.new("targets must be I32, not #{targets.dtype}") unless targets.dtype.i32?
      picks = targets.to_a(Int32).map_with_index do |target, row|
        raise ShapeError.new("target #{target} out of range for #{classes} classes") unless 0 <= target < classes
        row * classes + target
      end
      picked = log_softmax.reshape([rows * classes, 1]).gather(Tensor.from(picks, [rows], device))
      -picked.mean
    end

    # Returns layer normalisation over the last axis: each row is shifted to mean 0,
    # scaled to variance 1, then multiplied by *weight* and offset by *bias*, both of
    # shape `[last]`.
    #
    # With `F32` data and *weight* and *bias* of shape `[last]`, it is one fused op
    # (`Op::LayerNorm`), which computes what the composite computes, operation for
    # operation, on host devices.
    def layernorm(weight : Tensor, bias : Tensor, eps : Number = 1e-5) : Tensor
      if fusable_row?(-1) && weight.shape == [shape[-1]] && bias.shape == [shape[-1]]
        check_compatible("layernorm", {weight, bias})
        srcs = [self, weight.expand(shape), bias.expand(shape)]
        return Tensor.lazy(Node.new(Op::LayerNorm, srcs, shape, dtype, device, epsilon: eps.to_f32))
      end
      centred = self - mean(-1, keepdims: true)
      variance = (centred * centred).mean(-1, keepdims: true)
      centred / (variance + eps).sqrt * weight + bias
    end

    # Returns GELU with the tanh approximation used by GPT-2:
    # `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`.
    #
    # One fused kernel (`Tensomite::Kernels::GELU`), which computes exactly what
    # the composite `x * 0.5 * (((x + x * x * x * 0.044715) * √(2/π)).tanh + 1)`
    # computes, operation for operation.
    def gelu : Tensor
      Kernels::GELU.call(self)
    end

    # Returns an `F32` mask of shape `[rows, cols]` to add to attention scores:
    # 0 where query row `i` may attend to key column `j`, and -∞ where `j` lies in
    # its future. The last row sees every column, so with a KV cache holding earlier
    # positions, row `i` sees columns up to `i + cols - rows`.
    def self.causal_mask(rows : Int32, cols : Int32, device : Device = CPU.default) : Tensor
      raise ShapeError.new("causal mask needs cols >= rows, not #{rows}x#{cols}") unless cols >= rows
      past = cols - rows
      data = Array(Float32).new(rows * cols) do |idx|
        i, j = idx.divmod(cols)
        j > i + past ? -Float32::INFINITY : 0f32
      end
      from(data, [rows, cols], device)
    end

    # True when an op along *axis* can be fused: `F32` data, the last axis, and a
    # row with at least one element.
    private def fusable_row?(axis : Int32) : Bool
      dtype.f32? && ndim > 0 && view.normalize_axis(axis) == ndim - 1 && shape[-1] > 0
    end

    private def row_op(op : Op) : Tensor
      Tensor.lazy(Node.new(op, [self], shape, dtype, device))
    end
  end
end
