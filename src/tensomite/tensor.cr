module Tensomite
  # A lazy n-dimensional array: a `View` over the output of a graph `Node`.
  #
  # Operations build graph nodes and do no work; `#realize`, or reading data back
  # with `#to_a`, runs them. Shape and dtype errors are raised when the operation is
  # called, not when it is realised. View operations share the node and copy nothing.
  #
  # ```
  # a = Tensor.from([1, 2, 3, 4, 5, 6].map(&.to_f32), [2, 3])
  # b = Tensor.from([10, 20, 30].map(&.to_f32), [3])
  # (a + b).sum(0).to_a(Float32) # => [25.0, 47.0, 69.0]
  # ```
  struct Tensor
    getter node : Node
    getter view : View

    def initialize(@node, @view)
    end

    # Returns a realised tensor holding *data* in row-major order with the given *shape*.
    def self.from(data : Array(T), shape : Array(Int32), device : Device = CPU.default) : Tensor forall T
      unless data.size == shape.product(1)
        raise ShapeError.new("#{data.size} elements do not fill shape #{shape}")
      end
      dtype = DType.of(T)
      buffer = device.alloc(data.size * dtype.bytesize)
      device.upload(buffer, Slice.new(data.to_unsafe, data.size).to_unsafe_bytes)
      new(Node.new(buffer, shape, dtype, device), View.contiguous(shape))
    end

    def shape : Array(Int32)
      view.shape
    end

    def ndim : Int32
      view.ndim
    end

    def size : Int32
      view.size
    end

    def dtype : DType
      node.dtype
    end

    def device : Device
      node.device
    end

    # Returns the elementwise sum, broadcasting both operands to a common shape.
    def +(other : Tensor) : Tensor
      binary(Op::Add, other)
    end

    # Returns the elementwise product, broadcasting both operands to a common shape.
    def *(other : Tensor) : Tensor
      binary(Op::Mul, other)
    end

    # Returns the sum of all elements as a rank-0 tensor, or the sum over *axis*
    # with that dimension removed.
    def sum(axis : Int32? = nil) : Tensor
      shape = [] of Int32
      if axis
        axis = view.normalize_axis(axis)
        shape = self.shape.dup.tap(&.delete_at(axis))
      end
      Tensor.lazy(Node.new(Op::Sum, [self], shape, dtype, device, axis))
    end

    # Returns a tensor with dimensions reordered; see `View#permute`.
    def permute(axes : Array(Int32)) : Tensor
      Tensor.new(node, view.permute(axes))
    end

    # Returns a tensor with two dimensions swapped, by default the last two.
    def transpose(a : Int32 = -2, b : Int32 = -1) : Tensor
      Tensor.new(node, view.transpose(a, b))
    end

    # Returns a tensor broadcast to *shape* without copying.
    def expand(shape : Array(Int32)) : Tensor
      Tensor.new(node, view.expand(shape))
    end

    # Returns a tensor with a new shape and the same elements, copying only when
    # the current view is not contiguous.
    def reshape(shape : Array(Int32)) : Tensor
      if reshaped = view.reshape(shape)
        Tensor.new(node, reshaped)
      else
        contiguous.reshape(shape)
      end
    end

    # Returns a tensor whose data is laid out row-major, copying only when needed.
    def contiguous : Tensor
      return self if view.contiguous?
      Tensor.lazy(Node.new(Op::Copy, [self], shape, dtype, device))
    end

    # Computes this tensor and everything it depends on; returns `self`.
    def realize : Tensor
      Scheduler.realize(node)
      self
    end

    # Realises the tensor and returns its elements in row-major order.
    #
    # Raises `DTypeError` unless *type* matches `#dtype`.
    def to_a(type : T.class) : Array(T) forall T
      unless DType.of(T) == dtype
        raise DTypeError.new("cannot read #{dtype} tensor as #{T}")
      end
      realize
      buffer = node.buffer!
      bytes = device.map(buffer) || Bytes.new(buffer.bytesize).tap { |host| device.download(buffer, host) }
      xs = bytes.unsafe_slice_of(T)
      Array(T).new(size) { |i| xs[view.offset_of(i)] }
    end

    protected def self.lazy(node : Node) : Tensor
      new(node, View.contiguous(node.shape))
    end

    private def binary(op : Op, other : Tensor) : Tensor
      unless dtype == other.dtype
        raise DTypeError.new("#{op} of #{dtype} and #{other.dtype}")
      end
      unless device.same?(other.device)
        raise DeviceError.new("#{op} of tensors on different devices")
      end
      target = View.broadcast_shape(shape, other.shape)
      Tensor.lazy(Node.new(op, [expand(target), other.expand(target)], target, dtype, device))
    end
  end
end
