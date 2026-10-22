module Tensomite
  # A buffer held in host memory by a `HostDevice`.
  class HostBuffer < Buffer
    getter bytes : Bytes

    def initialize(device : Device, bytesize : Int32)
      super
      @bytes = Bytes.new(bytesize)
    end
  end

  # A device whose kernels run on the host CPU, in pure Crystal.
  #
  # Kernels are naive and run in a fixed order, so results are bit-identical from
  # run to run and independent of shape. Subclasses decide whether their memory is
  # host-visible through `#map`.
  abstract class HostDevice < Device
    def alloc(bytesize : Int32) : Buffer
      HostBuffer.new(self, bytesize)
    end

    def upload(buffer : Buffer, data : Bytes) : Nil
      data.copy_to(own(buffer).bytes)
    end

    def download(buffer : Buffer, data : Bytes) : Nil
      own(buffer).bytes.copy_to(data)
    end

    def execute(node : Node, output : Buffer) : Nil
      case node.dtype
      in .f32? then execute(Float32, node, output)
      in .i32? then execute(Int32, node, output)
      end
    end

    def sync : Nil
    end

    # Returns *buffer* as a `HostBuffer` of this device; raises `DeviceError` otherwise.
    protected def own(buffer : Buffer) : HostBuffer
      if buffer.device.same?(self) && (host = buffer.as?(HostBuffer))
        host
      else
        raise DeviceError.new("buffer does not belong to #{self.class}")
      end
    end

    private def execute(type : T.class, node : Node, output : Buffer) : Nil forall T
      dst = elements(T, output)
      case node.op
      in Op::Add     then binary(T, node, dst) { |x, y| x + y }
      in Op::Sub     then binary(T, node, dst) { |x, y| x - y }
      in Op::Mul     then binary(T, node, dst) { |x, y| x * y }
      in Op::Div     then binary(T, node, dst) { |x, y| T.new(x / y) }
      in Op::Maximum then binary(T, node, dst) { |x, y| x > y ? x : y }
      in Op::Exp     then unary(T, node, dst) { |x| T.new(Math.exp(x)) }
      in Op::Log     then unary(T, node, dst) { |x| T.new(Math.log(x)) }
      in Op::Tanh    then unary(T, node, dst) { |x| T.new(Math.tanh(x)) }
      in Op::Sqrt    then unary(T, node, dst) { |x| T.new(Math.sqrt(x)) }
      in Op::Sum     then reduce(T, node, dst, T.zero) { |acc, x| acc + x }
      in Op::Max     then reduce(T, node, dst, nil) { |acc, x| x > acc ? x : acc }
      in Op::Copy    then copy(T, node, dst)
      in Op::Cat     then cat(T, node, dst)
      in Op::Gather  then gather(T, node, dst)
      in Op::MatMul  then matmul(T, node, dst)
      in Nil         then raise Error.new("leaf nodes are created realised")
      end
    end

    # Applies the block to each pair of elements; sources already have the output shape.
    private def binary(type : T.class, node : Node, dst : Slice(T), & : T, T -> T) : Nil forall T
      a, b = node.srcs
      xs, ys = elements(T, a), elements(T, b)
      dst.size.times do |i|
        dst[i] = yield xs[a.view.offset_of(i)], ys[b.view.offset_of(i)]
      end
    end

    private def unary(type : T.class, node : Node, dst : Slice(T), & : T -> T) : Nil forall T
      src = node.srcs.first
      xs = elements(T, src)
      dst.size.times { |i| dst[i] = yield xs[src.view.offset_of(i)] }
    end

    # Folds each output's inputs left to right, starting from *init*, or from the
    # first input when *init* is `nil`.
    #
    # 1. Permutes the reduced axis to the end, so each output owns a run of `n`
    #    consecutive logical indices; with no axis, the single output owns them all.
    # 2. Folds each run in index order.
    private def reduce(type : T.class, node : Node, dst : Slice(T), init : T?, & : T, T -> T) : Nil forall T
      src = node.srcs.first
      xs = elements(T, src)
      view = src.view
      n = view.size
      if axis = node.axis
        view = view.permute((0...view.ndim).reject(axis) + [axis])
        n = view.shape.last
      end
      dst.size.times do |row|
        base = row * n
        acc, start = init.nil? ? {xs[view.offset_of(base)], 1} : {init, 0}
        (start...n).each { |j| acc = yield acc, xs[view.offset_of(base + j)] }
        dst[row] = acc
      end
    end

    private def copy(type : T.class, node : Node, dst : Slice(T)) : Nil forall T
      src = node.srcs.first
      xs = elements(T, src)
      dst.size.times { |i| dst[i] = xs[src.view.offset_of(i)] }
    end

    # Writes each source into its slice of the output, viewed along the axis.
    private def cat(type : T.class, node : Node, dst : Slice(T)) : Nil forall T
      axis = node.axis || raise Error.new("cat needs an axis")
      out_view = View.contiguous(node.shape)
      start = 0
      node.srcs.each do |src|
        xs = elements(T, src)
        stop = start + src.shape[axis]
        region = out_view.slice(axis, start, stop)
        src.size.times { |i| dst[region.offset_of(i)] = xs[src.view.offset_of(i)] }
        start = stop
      end
    end

    # Copies row `ids[i]` of the table into output row `i`.
    private def gather(type : T.class, node : Node, dst : Slice(T)) : Nil forall T
      table, ids = node.srcs
      rows, width = table.shape
      xs = elements(T, table)
      idx = elements(Int32, ids)
      ids.size.times do |i|
        id = idx[ids.view.offset_of(i)]
        unless 0 <= id < rows
          raise ExecutionError.new("index #{id} out of range for #{rows} rows")
        end
        width.times { |j| dst[i * width + j] = xs[table.view.offset_of(id * width + j)] }
      end
    end

    # Computes each output as a dot product over `k`, added in index order.
    private def matmul(type : T.class, node : Node, dst : Slice(T)) : Nil forall T
      a, b = node.srcs
      m, k = a.shape[-2], a.shape[-1]
      n = b.shape[-1]
      xs, ys = elements(T, a), elements(T, b)
      dst.size.times do |i|
        batch, rem = i.divmod(m * n)
        row, col = rem.divmod(n)
        acc = T.zero
        k.times do |step|
          acc += xs[a.view.offset_of((batch * m + row) * k + step)] * ys[b.view.offset_of((batch * k + step) * n + col)]
        end
        dst[i] = acc
      end
    end

    private def elements(type : T.class, tensor : Tensor) : Slice(T) forall T
      elements(T, tensor.node.buffer!)
    end

    private def elements(type : T.class, buffer : Buffer) : Slice(T) forall T
      own(buffer).bytes.unsafe_slice_of(T)
    end
  end
end
