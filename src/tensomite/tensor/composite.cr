module Tensomite
  struct Tensor
    # Returns the softmax along *axis*: `exp(x - max) / sum(exp(x - max))`.
    # Subtracting the maximum keeps `exp` from overflowing without changing the result.
    def softmax(axis : Int32 = -1) : Tensor
      e = (self - max(axis, keepdims: true)).exp
      e / e.sum(axis, keepdims: true)
    end

    # Returns layer normalisation over the last axis: each row is shifted to mean 0,
    # scaled to variance 1, then multiplied by *weight* and offset by *bias*, both of
    # shape `[last]`.
    def layernorm(weight : Tensor, bias : Tensor, eps : Number = 1e-5) : Tensor
      centred = self - mean(-1, keepdims: true)
      variance = (centred * centred).mean(-1, keepdims: true)
      centred / (variance + eps).sqrt * weight + bias
    end

    # Returns GELU with the tanh approximation used by GPT-2:
    # `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))`.
    def gelu : Tensor
      inner = (self + self * self * self * 0.044715) * Math.sqrt(2 / Math::PI)
      self * 0.5 * (inner.tanh + 1)
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
  end
end
