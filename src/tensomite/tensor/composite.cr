module Tensomite
  # How `Tensor#cross_entropy` combines its rows' losses: their mean, their
  # sum, or none, one loss per row.
  enum Reduction
    Mean
    Sum
    None
  end

  struct Tensor
    # Returns the softmax along *axis*: `exp(x - max) / sum(exp(x - max))`.
    # Subtracting the maximum keeps `exp` from overflowing without changing the result.
    #
    # With *scale* or *mask*, `x` is this tensor times *scale*, plus *mask*
    # broadcast to its shape, as attention scores are:
    #
    # ```
    # scores.softmax(scale: 1 / Math.sqrt(head_width), mask: Tensor.causal_mask(n, n))
    # ```
    #
    # Along the last axis it is one fused op (`Op::Softmax`), with the scale and
    # mask applied as it reads each element, and its gradient is another
    # (`Op::SoftmaxGrad`); both compute what the composite computes, operation
    # for operation, so they give its bits.
    def softmax(axis : Int32 = -1, *, scale : Number? = nil, mask : Tensor? = nil) : Tensor
      if fusable_row?(axis)
        if mask.nil? || View.broadcast_shape(shape, mask.shape) == shape
          check_compatible("softmax", {mask}) if mask
          srcs = mask ? [self, mask.expand(shape)] : [self]
          return Tensor.lazy(Node.new(Op::Softmax, srcs, shape, dtype, device, scale: scale.try(&.to_f32)))
        end
      end
      x = scale ? self * scale : self
      x = x + mask if mask
      return x.softmax(axis) if scale || mask
      e = (x - x.max(axis, keepdims: true)).exp
      e / e.sum(axis, keepdims: true)
    end

    # Returns the log of the softmax along *axis*: `x - max - log(sum(exp(x - max)))`.
    # Fused along the last axis, as `#softmax` is.
    def log_softmax(axis : Int32 = -1) : Tensor
      return row_op(Op::LogSoftmax) if fusable_row?(axis)
      shifted = self - max(axis, keepdims: true)
      shifted - shifted.exp.sum(axis, keepdims: true).log
    end

    # Returns the cross-entropy of these `[n, classes]` logits against *targets*,
    # one class index per row: each row's loss is `-log_softmax[row, target]`.
    #
    # *reduction* gives their mean (the default), their sum, or, with `:none`,
    # each row's loss as an `[n]` tensor, so that the caller can weight them:
    # `(logits.cross_entropy(targets, reduction: :none) * weights).sum / n`.
    # Rows whose target is *ignore_index* add nothing: they are left out of the
    # sum and the mean's count, and their loss is 0 with `:none`. With every
    # row ignored, the mean is NaN.
    #
    # The targets are read on the host to pick each row's entry, so *targets* is
    # realised when this is called.
    #
    # ```
    # logits.cross_entropy(targets)                   # the mean, a scalar
    # logits.cross_entropy(targets, reduction: :none) # [n]
    # logits.cross_entropy(targets, ignore_index: -1) # the mean over rows not -1
    # ```
    def cross_entropy(targets : Tensor, reduction : Reduction = :mean, ignore_index : Int32? = nil) : Tensor
      raise ShapeError.new("cross_entropy needs [n, classes] logits, not #{shape}") unless ndim == 2
      rows, classes = shape
      ids = target_ids(targets, rows, classes, ignore_index)
      flat = log_softmax.reshape([rows * classes, 1])
      return row_losses(flat, ids, classes, ignore_index) if reduction.none?
      picks = [] of Int32
      ids.each_with_index { |target, row| picks << row * classes + target unless target == ignore_index }
      if picks.empty?
        return Tensor.from([reduction.mean? ? Float32::NAN : 0f32], [] of Int32, device)
      end
      picked = flat.gather(Tensor.from(picks, [picks.size], device))
      reduction.mean? ? -picked.mean : -picked.sum
    end

    # Returns *targets* read on the host, after checking that they are `I32`,
    # one per row, and each a class index or *ignore_index*.
    private def target_ids(targets : Tensor, rows : Int32, classes : Int32, ignore_index : Int32?) : Array(Int32)
      raise ShapeError.new("#{targets.size} targets for #{rows} rows") unless targets.size == rows
      raise DTypeError.new("targets must be I32, not #{targets.dtype}") unless targets.dtype.i32?
      ids = targets.to_a(Int32)
      ids.each do |target|
        next if target == ignore_index || 0 <= target < classes
        raise ShapeError.new("target #{target} out of range for #{classes} classes")
      end
      ids
    end

    # Returns each row's `-log_softmax[row, target]` from *flat*, the
    # log-softmax as one column, and 0 for rows whose target is *ignore_index*.
    private def row_losses(flat : Tensor, ids : Array(Int32), classes : Int32, ignore_index : Int32?) : Tensor
      rows = ids.size
      picks = ids.map_with_index { |target, row| row * classes + (target == ignore_index ? 0 : target) }
      losses = -flat.gather(Tensor.from(picks, [rows], device)).reshape([rows])
      return losses unless ignore_index && ids.includes?(ignore_index)
      losses * Tensor.from(ids.map { |target| target == ignore_index ? 0f32 : 1f32 }, [rows], device)
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

    # Returns SiLU, `x · sigmoid(x)`, computed as `x / (e^-x + 1)` in one fused
    # kernel (`Tensomite::Kernels::SILU`).
    def silu : Tensor
      Kernels::SILU.call(self)
    end

    # Returns `silu(self) · up`, the gate of a SwiGLU MLP, in one fused kernel
    # (`Tensomite::Kernels::SWIGLU`); *up* has this tensor's shape.
    def swiglu(up : Tensor) : Tensor
      Kernels::SWIGLU.call(self, up)
    end

    # Returns root-mean-square normalisation of the last axis, scaled by
    # *weight*, of shape `[last]`: `x / √(mean(x²) + eps) · weight`, as Llama
    # normalises.
    #
    # With `F32` data and *weight* of shape `[last]`, it is one fused op
    # (`Op::RMSNorm`), which computes what the composite computes, operation for
    # operation, on host devices.
    def rmsnorm(weight : Tensor, eps : Number = 1e-5) : Tensor
      if fusable_row?(-1) && weight.shape == [shape[-1]]
        check_compatible("rmsnorm", {weight})
        return Tensor.lazy(Node.new(Op::RMSNorm, [self, weight.expand(shape)], shape, dtype, device, epsilon: eps.to_f32))
      end
      self / ((self * self).mean(-1, keepdims: true) + eps).sqrt * weight
    end

    # The most query rows per head, cached positions, and value columns that
    # `.attention` fuses.
    ATTENTION_ROWS      =    8
    ATTENTION_POSITIONS = 4096
    ATTENTION_COLUMNS   =  128

    # Returns attention: `softmax(q · kᵀ · scale + mask) · v`, for queries *q*
    # `[..., rows, width]`, keys *k* `[..., positions, width]` and values *v*
    # `[..., positions, value width]`, with *mask* broadcast to `[...,
    # rows, positions]`.
    #
    # On a device that `Device#fuses_attention?`, with `F32` data, at most
    # `ATTENTION_ROWS` rows per head, `ATTENTION_POSITIONS` positions and
    # `ATTENTION_COLUMNS` value columns, as decoding has, it is one op
    # (`Op::Attention`), whose scores never leave the GPU; it adds in the
    # composite's orders, so on a shape-invariant device it has the
    # composite's bits. Otherwise it is the composite: a matmul, a fused
    # softmax and a matmul.
    def self.attention(q : Tensor, k : Tensor, v : Tensor, *, scale : Number, mask : Tensor? = nil) : Tensor
      if attention_fusable?(q, k, v, mask)
        q.check_compatible("attention", mask ? [k, v, mask] : [k, v])
        scores = q.shape[0...-1] + [k.shape[-2]]
        srcs = [q, k, v]
        srcs << mask.expand(scores) if mask
        return Tensor.lazy(Node.new(Op::Attention, srcs, q.shape[0...-1] + [v.shape[-1]], q.dtype, q.device, scale: scale.to_f32))
      end
      q.matmul(k.transpose).softmax(scale: scale, mask: mask).matmul(v)
    end

    # Returns true when `.attention` of these makes one `Op::Attention`.
    private def self.attention_fusable?(q : Tensor, k : Tensor, v : Tensor, mask : Tensor?) : Bool
      return false unless q.device.fuses_attention? && {q, k, v}.all?(&.dtype.f32?) && mask.try(&.dtype.f32?) != false
      return false unless attention_shapes?(q, k, v)
      return false unless q.shape[-2] <= ATTENTION_ROWS && 0 < k.shape[-2] <= ATTENTION_POSITIONS && v.shape[-1] <= ATTENTION_COLUMNS
      scores = q.shape[0...-1] + [k.shape[-2]]
      mask.nil? || View.broadcast_shape(scores, mask.shape) == scores
    end

    # Returns true when *q*, *k* and *v* are `[..., rows, width]`, `[...,
    # positions, width]` and `[..., positions, value width]` with the same
    # leading dimensions.
    private def self.attention_shapes?(q : Tensor, k : Tensor, v : Tensor) : Bool
      return false unless q.ndim >= 2 && q.ndim == k.ndim && k.ndim == v.ndim
      lead = q.shape[0...-2]
      lead == k.shape[0...-2] && lead == v.shape[0...-2] && q.shape[-1] == k.shape[-1] && k.shape[-2] == v.shape[-2]
    end

    # Returns `self · cos + rotate_half(self) · sin`, rotary embeddings' rotation
    # of the pairs `(i, i + width/2)` of the last axis, for *cos* and *sin* that
    # broadcast to this shape.
    #
    # With `F32` data it is one fused op (`Op::Rope`), which computes what the
    # composite computes, operation for operation, on every device: it has only
    # multiplications, a negation and an addition, each exact in IEEE arithmetic.
    def rope(cos : Tensor, sin : Tensor) : Tensor
      if fusable_row?(-1) && shape[-1].even? && View.broadcast_shape(shape, cos.shape) == shape &&
         View.broadcast_shape(shape, sin.shape) == shape
        check_compatible("rope", {cos, sin})
        return Tensor.lazy(Node.new(Op::Rope, [self, cos.expand(shape), sin.expand(shape)], shape, dtype, device))
      end
      self * cos + rotate_half * sin
    end

    # Returns the last axis's two halves swapped, the first negated:
    # `[x₁, x₂] → [-x₂, x₁]`, as rotary embeddings rotate pairs of features.
    def rotate_half : Tensor
      half = shape[-1] // 2
      raise ShapeError.new("rotate_half needs an even last axis, not #{shape[-1]}") unless shape[-1] == 2 * half
      Tensor.cat([-slice(-1, half, 2 * half), slice(-1, 0, half)], -1)
    end

    # Returns the transpose of `rotate_half`, `[y₁, y₂] → [y₂, -y₁]`, which carries
    # a gradient back through it.
    protected def rotate_half_transposed : Tensor
      half = shape[-1] // 2
      Tensor.cat([slice(-1, half, 2 * half), -slice(-1, 0, half)], -1)
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
