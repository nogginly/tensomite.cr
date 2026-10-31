module Tensomite
  struct Tensor
    # Returns the elementwise sum, broadcasting both operands to a common shape.
    def +(other : Tensor) : Tensor
      binary(Op::Add, other)
    end

    # Returns the elementwise difference, broadcasting both operands to a common shape.
    def -(other : Tensor) : Tensor
      binary(Op::Sub, other)
    end

    # Returns the elementwise product, broadcasting both operands to a common shape.
    def *(other : Tensor) : Tensor
      binary(Op::Mul, other)
    end

    # Returns the elementwise quotient, broadcasting both operands to a common shape.
    # Float dtypes only.
    def /(other : Tensor) : Tensor
      check_float("/")
      binary(Op::Div, other)
    end

    # Returns the elementwise larger value, broadcasting both operands to a common shape.
    def maximum(other : Tensor) : Tensor
      binary(Op::Maximum, other)
    end

    # Returns 1 where this tensor is at least *other* and 0 elsewhere, in this
    # tensor's dtype, broadcasting both operands to a common shape.
    def greater_equal(other : Tensor) : Tensor
      binary(Op::GreaterEqual, other)
    end

    # Applies the operator with *other* broadcast as a scalar of this tensor's dtype.
    def +(other : Number) : Tensor
      self + scalar(other)
    end

    # :ditto:
    def -(other : Number) : Tensor
      self - scalar(other)
    end

    # :ditto:
    def *(other : Number) : Tensor
      self * scalar(other)
    end

    # :ditto:
    def /(other : Number) : Tensor
      self / scalar(other)
    end

    # Returns the elementwise negation.
    def - : Tensor
      self * -1
    end

    # Returns `e^x` elementwise. Float dtypes only.
    def exp : Tensor
      unary(Op::Exp)
    end

    # Returns the natural logarithm elementwise. Float dtypes only.
    def log : Tensor
      unary(Op::Log)
    end

    # Returns the hyperbolic tangent elementwise. Float dtypes only.
    def tanh : Tensor
      unary(Op::Tanh)
    end

    # Returns the square root elementwise. Float dtypes only.
    def sqrt : Tensor
      unary(Op::Sqrt)
    end

    # Returns *kernel* applied to *inputs*, broadcast to a common shape. Call it as
    # `kernel.call(inputs)`; see `Tensomite.kernel`.
    def self.kernel(kernel : Kernel, inputs : Array(Tensor)) : Tensor
      unless inputs.size == kernel.arity
        raise ArgumentError.new("kernel #{kernel.name} takes #{kernel.arity} inputs, not #{inputs.size}")
      end
      first = inputs.first
      first.check_compatible("kernel #{kernel.name}", inputs)
      first.check_float("kernel #{kernel.name}")
      raise DTypeError.new("kernel #{kernel.name} needs F32, not #{first.dtype}") unless first.dtype.f32?
      target = inputs.map(&.shape).reduce { |a, b| View.broadcast_shape(a, b) }
      Tensor.lazy(Node.new(Op::Kernel, inputs.map(&.expand(target)), target, first.dtype, first.device, kernel: kernel))
    end

    private def binary(op : Op, other : Tensor) : Tensor
      check_compatible(op.to_s, {other})
      target = View.broadcast_shape(shape, other.shape)
      Tensor.lazy(Node.new(op, [expand(target), other.expand(target)], target, dtype, device))
    end

    private def unary(op : Op) : Tensor
      check_float(op.to_s)
      Tensor.lazy(Node.new(op, [self], shape, dtype, device))
    end

    # Constants that arithmetic with a number has made, by device, dtype and
    # value, so that a training step uploads each one once rather than once per
    # use. Cleared when it reaches `SCALARS` entries, as a learning-rate
    # schedule makes new values every step.
    @@scalars = {} of {UInt64, DType, Int64} => Tensor

    private SCALARS = 4096

    # Returns a scalar of *value* in this tensor's dtype on its device, shared
    # with every other use of the same value there.
    private def scalar(value : Number) : Tensor
      return Tensor.scalar(value, dtype, device) unless dtype.f32? || value.to_i64 == value
      bits = dtype.f32? ? value.to_f32.unsafe_as(UInt32).to_i64 : value.to_i64
      key = {device.object_id, dtype, bits}
      @@scalars[key]? || begin
        @@scalars.clear if @@scalars.size >= SCALARS
        @@scalars[key] = Tensor.scalar(value, dtype, device)
      end
    end
  end
end
