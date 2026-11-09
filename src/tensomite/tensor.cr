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

    # Returns a tensor showing *node* through *view*.
    #
    # Raises `DTypeError` for a view of a block-format node that is not a run
    # of its whole rows, laid out as they are stored (Decision N).
    def initialize(@node, @view)
      check_block_view if node.dtype.block?
    end

    private def check_block_view : Nil
      row = node.shape.last? || 0
      unless view.strides == View.row_major_strides(view.shape) && view.shape.last? == row && view.offset % Math.max(row, 1) == 0
        raise DTypeError.new("a #{node.dtype} tensor can only be viewed as runs of its whole rows, not as #{view.shape}")
      end
    end

    # Returns a realised tensor holding *data* in row-major order with the given *shape*.
    def self.from(data : Array(T), shape : Array(Int32), device : Device = CPU.default) : Tensor forall T
      unless data.size == shape.product(1)
        raise ShapeError.new("#{data.size} elements do not fill shape #{shape}")
      end
      dtype = DType.of(T)
      buffer = device.alloc(dtype.bytes_for(data.size))
      device.upload(buffer, Slice.new(data.to_unsafe, data.size).to_unsafe_bytes)
      new(Node.new(buffer, shape, dtype, device), View.contiguous(shape))
    end

    # Returns a realised rank-0 tensor holding *value* converted to *dtype*.
    #
    # Raises `DTypeError` when *value* has a fractional part and *dtype* is an integer type.
    def self.scalar(value : Number, dtype : DType, device : Device = CPU.default) : Tensor
      case dtype
      in .f32? then from([value.to_f32], [] of Int32, device)
      in .i32?
        raise DTypeError.new("#{value} is not an integer") unless value.to_i32 == value
        from([value.to_i32], [] of Int32, device)
      in .q8_0?, .q4_0?, .q4_1?
        raise DTypeError.new("cannot make a #{dtype} scalar; quantize an F32 tensor instead")
      end
    end

    # Returns a realised tensor of *shape* with every element *value*.
    def self.full(shape : Array(Int32), value : Number, dtype : DType = DType::F32, device : Device = CPU.default) : Tensor
      case dtype
      in .f32? then from(Array.new(shape.product(1), value.to_f32), shape, device)
      in .i32? then from(Array.new(shape.product(1), value.to_i32), shape, device)
      in .q8_0?, .q4_0?, .q4_1?
        full(shape, value, DType::F32, device).quantize(dtype)
      end
    end

    # Returns a realised tensor of *shape* filled with zeros.
    def self.zeros(shape : Array(Int32), dtype : DType = DType::F32, device : Device = CPU.default) : Tensor
      full(shape, 0, dtype, device)
    end

    def shape : Array(Int32)
      view.shape
    end

    # Names the node this tensor views, for diagrams, and returns `self`.
    def named(name : String) : Tensor
      node.name = name
      self
    end

    # Returns the name of the node this tensor views, if it has one.
    def name : String?
      node.name
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

    # Returns the indices `start...stop` along *axis* without copying.
    def slice(axis : Int32, start : Int32, stop : Int32) : Tensor
      Tensor.new(node, view.slice(axis, start, stop))
    end

    # Returns a tensor with a new shape and the same elements, copying only when
    # no view can give it (see `View#reshape`).
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

    # Realises this tensor and returns it as a leaf over the same buffer, so later
    # graphs, and `Tensomite.grad`, treat it as a constant rather than as the result
    # of the ops that made it.
    def detach : Tensor
      realize
      Tensor.new(Node.new(node.buffer!, node.shape, dtype, device), view)
    end

    # Computes this tensor and everything it depends on; returns `self`.
    def realize : Tensor
      Scheduler.realize(node)
      self
    end

    # Computes *tensors* and everything they depend on, waiting for the device
    # once rather than once per tensor; returns *tensors*.
    #
    # ```
    # Tensor.realize([loss, *updated_params])
    # ```
    #
    # With *release*, the intermediate tensors computed on the way give their
    # memory back as soon as nothing more in this call reads them, and can no
    # longer be read: use it when only *tensors* are wanted, as in a training
    # step (see `Scheduler.realize`).
    def self.realize(tensors : Array(Tensor), release : Bool = false) : Array(Tensor)
      Scheduler.realize(tensors.map(&.node), release)
      tensors
    end

    # Realises the tensor and returns its elements in row-major order.
    #
    # Raises `DTypeError` unless *type* matches `#dtype`.
    def to_a(type : T.class) : Array(T) forall T
      unless DType.of(T) == dtype
        raise DTypeError.new("cannot read #{dtype} tensor as #{T}")
      end
      # Allocates the result first: once `host_bytes` is taken, nothing may
      # allocate, or a collection could recycle the memory while it is read.
      result = Array(T).new(size, T.zero)
      xs = realize.host_bytes.unsafe_slice_of(T)
      size.times { |i| result[i] = xs[view.offset_of(i)] }
      result
    end

    # Returns a realised copy of this tensor on *device*, or `self` when it is already there.
    #
    # The copy goes through host memory: download, or map, from the source device,
    # then upload to *device*.
    def to(device : Device) : Tensor
      return self if device.same?(self.device)
      return Tensor.from_blocks(block_bytes, shape, dtype, device) if dtype.block?
      src = contiguous.realize
      buffer = device.alloc(dtype.bytes_for(src.size))
      device.upload(buffer, src.host_bytes[0, buffer.bytesize])
      Tensor.new(Node.new(buffer, shape, dtype, device), View.contiguous(shape))
    end

    protected def self.lazy(node : Node) : Tensor
      new(node, View.contiguous(node.shape))
    end

    # Returns the realised node's bytes in host memory, mapped when the device allows it.
    protected def host_bytes : Bytes
      buffer = node.buffer!
      device.map(buffer) || Bytes.new(buffer.bytesize).tap { |host| device.download(buffer, host) }
    end

    # Returns the gradient of the node this tensor views, from *grad* of this
    # tensor's shape: the sum, at each position of the node, of the elements of
    # *grad* that the view reads from it.
    protected def unview(grad : Tensor) : Tensor
      Tensor.lazy(Node.new(Op::Unview, [grad], node.shape, dtype, device, source_view: view))
    end

    # Returns a `[rows, width]` tensor where row `ids[i]` accumulates row `i` of this
    # `[n, width]` tensor.
    protected def scatter_add_rows(ids : Tensor, rows : Int32) : Tensor
      Tensor.lazy(Node.new(Op::ScatterAdd, [self, ids], [rows, shape[1]], dtype, device))
    end

    # Raises unless *others* share this tensor's dtype and device.
    protected def check_compatible(name : String, others : Enumerable(Tensor)) : Nil
      others.each do |other|
        unless dtype == other.dtype
          raise DTypeError.new("#{name} of #{dtype} and #{other.dtype}")
        end
        unless device.same?(other.device)
          raise DeviceError.new("#{name} of tensors on different devices")
        end
      end
    end

    # Raises `DTypeError` unless the dtype is a float type.
    protected def check_float(name : String) : Nil
      raise DTypeError.new("#{name} needs a float dtype, not #{dtype}") unless dtype.float?
    end
  end
end
