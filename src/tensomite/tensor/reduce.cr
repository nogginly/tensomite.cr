module Tensomite
  struct Tensor
    # Returns the sum of all elements as a rank-0 tensor, or the sum over *axis*.
    # The reduced axis is removed unless *keepdims* is true, when it is kept with size 1.
    def sum(axis : Int32? = nil, keepdims : Bool = false) : Tensor
      reduce(Op::Sum, axis, keepdims)
    end

    # Returns the largest element, overall or over *axis*; see `#sum` for *keepdims*.
    #
    # Raises `ShapeError` when there are no elements to compare.
    def max(axis : Int32? = nil, keepdims : Bool = false) : Tensor
      count = axis ? shape[view.normalize_axis(axis)] : size
      raise ShapeError.new("max over an empty axis of #{shape}") if count == 0
      reduce(Op::Max, axis, keepdims)
    end

    # Returns the arithmetic mean, overall or over *axis*; see `#sum` for *keepdims*.
    # Float dtypes only.
    def mean(axis : Int32? = nil, keepdims : Bool = false) : Tensor
      check_float("mean")
      count = axis ? shape[view.normalize_axis(axis)] : size
      sum(axis, keepdims) / count
    end

    private def reduce(op : Op, axis : Int32?, keepdims : Bool) : Tensor
      axis = view.normalize_axis(axis) if axis
      reduced = axis ? shape.dup.tap(&.delete_at(axis)) : [] of Int32
      result = Tensor.lazy(Node.new(op, [self], reduced, dtype, device, axis))
      return result unless keepdims
      kept = axis ? shape.dup.tap { |dims| dims[axis] = 1 } : Array.new(ndim, 1)
      result.reshape(kept)
    end
  end
end
