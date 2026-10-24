module Tensomite
  # A buffer held in host memory by a `HostDevice`.
  class HostBuffer < Buffer
    getter bytes : Bytes
    @owner : MappedFile? = nil

    def initialize(device : Device, bytesize : Int32)
      super
      @bytes = Bytes.new(bytesize)
    end

    # Wraps part of a file mapping without copying; the buffer keeps *owner* alive.
    def initialize(device : Device, @bytes : Bytes, @owner : MappedFile)
      super(device, @bytes.size)
    end

    # Wraps *bytes*, memory allocated for *device* or taken from its pool, or the
    # host-visible memory of a subclass's buffer, such as a GPU buffer.
    def initialize(device : Device, @bytes : Bytes)
      super(device, @bytes.size)
    end

    # Gives the memory back to the device's pool, unless it belongs to a file mapping.
    def finalize
      return if @owner
      device.as?(HostDevice).try(&.recycle(@bytes))
    end
  end

  # A device whose kernels run on the host CPU, in pure Crystal.
  #
  # Kernels are naive and run in a fixed order, so results are bit-identical from
  # run to run and independent of shape. Subclasses decide whether their memory is
  # host-visible through `#map`.
  abstract class HostDevice < Device
    # Freed buffers kept for reuse; see `BufferPool`. Holds up to 1 GiB by default.
    getter pool = BufferPool(Bytes).new(1_i64 << 30)

    # Returns a buffer from the pool when one of *bytesize* is free, otherwise a
    # new one. Its contents are undefined: kernels write every element.
    def alloc(bytesize : Int32) : Buffer
      HostBuffer.new(self, pool.take(bytesize) || Bytes.new(bytesize))
    end

    # Takes back the memory of a collected buffer.
    protected def recycle(bytes : Bytes) : Nil
      pool.give(bytes, bytes.size)
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

    # Host kernels add in index order whatever the shape.
    def shape_invariant? : Bool
      true
    end

    # Returns a buffer over *bytes* without copying; see `HostBuffer`.
    def adopt(bytes : Bytes, owner : MappedFile) : Buffer
      HostBuffer.new(self, bytes, owner)
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
      in Op::Add          then binary(T, node, dst) { |x, y| x + y }
      in Op::Sub          then binary(T, node, dst) { |x, y| x - y }
      in Op::Mul          then binary(T, node, dst) { |x, y| x * y }
      in Op::Div          then binary(T, node, dst) { |x, y| T.new(x / y) }
      in Op::Maximum      then binary(T, node, dst) { |x, y| x > y ? x : y }
      in Op::Exp          then unary(T, node, dst) { |x| T.new(Maths.exp(x.to_f32)) }
      in Op::Log          then unary(T, node, dst) { |x| T.new(Maths.log(x.to_f32)) }
      in Op::Tanh         then unary(T, node, dst) { |x| T.new(Maths.tanh(x.to_f32)) }
      in Op::Sqrt         then unary(T, node, dst) { |x| T.new(Math.sqrt(x)) }
      in Op::Sum          then reduce(T, node, dst, T.zero) { |acc, x| acc + x }
      in Op::Max          then reduce(T, node, dst, nil) { |acc, x| x > acc ? x : acc }
      in Op::Copy         then copy(T, node, dst)
      in Op::Cat          then cat(T, node, dst)
      in Op::Gather       then gather(T, node, dst)
      in Op::MatMul       then matmul(T, node, dst)
      in Op::GreaterEqual then binary(T, node, dst) { |x, y| x >= y ? T.new(1) : T.zero }
      in Op::Unview       then unview(T, node, dst)
      in Op::ScatterAdd   then scatter_add(T, node, dst)
      in Op::Softmax, Op::LogSoftmax, Op::LayerNorm
        rows(node, Slice.new(dst.to_unsafe.as(Float32*), dst.size))
      in Op::Kernel
        kernel = node.kernel || raise Error.new("kernel node without a kernel")
        kernel.host(Slice.new(dst.to_unsafe.as(Float32*), dst.size), node.srcs.map { |src| elements(Float32, src) }, node.srcs.map(&.view))
      in Nil then raise Error.new("leaf nodes are created realised")
      end
    end

    # Applies the block to each pair of elements; sources already have the output shape.
    #
    # Walks the output in runs (see `View.run_length`), finding each source's position
    # through its view once per run and stepping by its stride within it. Each run's
    # extent is checked once and then read through pointers, so that unit-stride
    # runs vectorise.
    private def binary(type : T.class, node : Node, dst : Slice(T), & : T, T -> T) : Nil forall T
      return if dst.empty?
      a, b = node.srcs
      xs, ys = elements(T, a), elements(T, b)
      run = View.run_length(dst.size, {a.view, b.view})
      sa, sb = a.view.run_stride, b.view.run_stride
      (dst.size // run).times do |run_index|
        base = run_index * run
        pa, pb = a.view.offset_of(base), b.view.offset_of(base)
        raise IndexError.new unless 0 <= pa && 0 <= pa + (run - 1) * sa < xs.size
        raise IndexError.new unless 0 <= pb && 0 <= pb + (run - 1) * sb < ys.size
        left, right, output = xs.to_unsafe + pa, ys.to_unsafe + pb, dst.to_unsafe + base
        if sa == 1 && sb == 1
          run.times { |j| output[j] = yield left[j], right[j] }
        elsif sa == 1 && sb == 0
          scalar = right[0]
          run.times { |j| output[j] = yield left[j], scalar }
        else
          run.times { |j| output[j] = yield left[j * sa], right[j * sb] }
        end
      end
    end

    # Applies the block to each element, walking in runs as `#binary` does.
    private def unary(type : T.class, node : Node, dst : Slice(T), & : T -> T) : Nil forall T
      return if dst.empty?
      src = node.srcs.first
      xs = elements(T, src)
      run = View.run_length(dst.size, {src.view})
      stride = src.view.run_stride
      (dst.size // run).times do |run_index|
        base = run_index * run
        pos = src.view.offset_of(base)
        raise IndexError.new unless 0 <= pos && 0 <= pos + (run - 1) * stride < xs.size
        input, output = xs.to_unsafe + pos, dst.to_unsafe + base
        if stride == 1
          run.times { |j| output[j] = yield input[j] }
        else
          run.times { |j| output[j] = yield input[j * stride] }
        end
      end
    end

    # Folds each output's inputs left to right, starting from *init*, or from the
    # first input when *init* is `nil`.
    #
    # 1. Permutes the reduced axis to the end, so each output owns a run of `n`
    #    consecutive logical indices; with no axis, the single output owns them all.
    # 2. Folds each output's indices in order, walking them in runs (see
    #    `#run_length`).
    private def reduce(type : T.class, node : Node, dst : Slice(T), init : T?, & : T, T -> T) : Nil forall T
      src = node.srcs.first
      xs = elements(T, src)
      view = src.view
      n = view.size
      if axis = node.axis
        view = view.permute((0...view.ndim).reject(axis) + [axis])
        n = view.shape.last
      end
      if n == 0
        dst.fill(init || T.zero)
        return
      end
      run = View.run_length(n, {view})
      stride = view.run_stride
      dst.size.times do |row|
        acc = init || xs[view.offset_of(row * n)]
        (n // run).times do |run_index|
          pos = view.offset_of(row * n + run_index * run)
          start = run_index == 0 && init.nil? ? 1 : 0
          (start...run).each { |j| acc = yield acc, xs[pos + j * stride] }
        end
        dst[row] = acc
      end
    end

    private def copy(type : T.class, node : Node, dst : Slice(T)) : Nil forall T
      unary(T, node, dst) { |x| x }
    end

    # Adds each source element into the output position the inverted view reads it
    # from, in index order, walking in runs as `#binary` does.
    private def unview(type : T.class, node : Node, dst : Slice(T)) : Nil forall T
      src = node.srcs.first
      view = node.source_view || raise Error.new("unview needs a source view")
      xs = elements(T, src)
      dst.fill(T.zero)
      return if src.size == 0
      run = View.run_length(src.size, {view, src.view})
      sv, ss = view.run_stride, src.view.run_stride
      (src.size // run).times do |run_index|
        base = run_index * run
        pv, ps = view.offset_of(base), src.view.offset_of(base)
        raise IndexError.new unless 0 <= pv && 0 <= pv + (run - 1) * sv < dst.size
        raise IndexError.new unless 0 <= ps && 0 <= ps + (run - 1) * ss < xs.size
        output, input = dst.to_unsafe + pv, xs.to_unsafe + ps
        if sv == 1 && ss == 1
          run.times { |j| output[j] += input[j] }
        else
          run.times { |j| output[j * sv] += input[j * ss] }
        end
      end
    end

    # The fused row ops, one row of the last axis at a time. Each does the
    # composite's operations in the composite's order, rounding each to `Float32`,
    # so it gives the composite's bits:
    #
    # - Softmax: the maximum, folded from the first element; then `e = exp(x - max)`
    #   and their sum, folded from zero; then `e / sum`.
    # - LogSoftmax: `shifted = x - max`; the sum of `exp(shifted)`; then
    #   `shifted - log(sum)`.
    # - LayerNorm: `mean = sum / n`; `centred = x - mean`; `variance =
    #   sum(centred²) / n`; then `centred / sqrt(variance + epsilon) * weight + bias`.
    #
    # A row lies along the last axis, so it is one run (see `View.run_length`): its
    # elements are found from the row's start and the view's stride.
    private def rows(node : Node, dst : Slice(Float32)) : Nil
      src = node.srcs.first
      n = node.shape.last
      xs, stride = elements(Float32, src), src.view.run_stride
      count = n.to_f32
      (dst.size // n).times do |row|
        base = row * n
        pos = src.view.offset_of(base)
        case node.op
        when Op::Softmax, Op::LogSoftmax
          raise IndexError.new unless 0 <= pos && 0 <= pos + (n - 1) * stride < xs.size
          row_in, row_out = xs.to_unsafe + pos, dst.to_unsafe + base
          top = row_in[0]
          (1...n).each { |j| top = (v = row_in[j * stride]) > top ? v : top }
          total = 0f32
          if node.op == Op::Softmax
            if stride == 1
              n.times { |j| row_out[j] = Maths.exp(row_in[j] - top) }
            else
              n.times { |j| row_out[j] = Maths.exp(row_in[j * stride] - top) }
            end
            n.times { |j| total += row_out[j] }
            n.times { |j| row_out[j] = row_out[j] / total }
          else
            n.times { |j| row_out[j] = row_in[j * stride] - top }
            n.times { |j| total += Maths.exp(row_out[j]) }
            log_total = Maths.log(total)
            n.times { |j| row_out[j] = row_out[j] - log_total }
          end
        else
          weight, bias = node.srcs[1], node.srcs[2]
          ws, bs = elements(Float32, weight), elements(Float32, bias)
          pw, sw = weight.view.offset_of(base), weight.view.run_stride
          pb, sb = bias.view.offset_of(base), bias.view.run_stride
          epsilon = node.epsilon || raise Error.new("layernorm needs an epsilon")
          total = 0f32
          n.times { |j| total += xs[pos + j * stride] }
          mean = total / count
          squares = 0f32
          n.times { |j| squares += (centred = dst[base + j] = xs[pos + j * stride] - mean) * centred }
          std = Math.sqrt(squares / count + epsilon)
          n.times { |j| dst[base + j] = dst[base + j] / std * ws[pw + j * sw] + bs[pb + j * sb] }
        end
      end
    end

    # Adds source row `i` into output row `ids[i]`, in index order.
    private def scatter_add(type : T.class, node : Node, dst : Slice(T)) : Nil forall T
      rows_src, ids = node.srcs
      rows, width = node.shape
      xs = elements(T, rows_src)
      idx = elements(Int32, ids)
      dst.fill(T.zero)
      ids.size.times do |i|
        id = idx[ids.view.offset_of(i)]
        raise ExecutionError.new("index #{id} out of range for #{rows} rows") unless 0 <= id < rows
        pos, stride = rows_src.view.offset_of(i * width), rows_src.view.run_stride
        width.times { |j| dst[id * width + j] += xs[pos + j * stride] }
      end
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
        if src.size > 0
          run = View.run_length(src.size, {region, src.view})
          sr, ss = region.run_stride, src.view.run_stride
          (src.size // run).times do |run_index|
            base = run_index * run
            pr, ps = region.offset_of(base), src.view.offset_of(base)
            run.times { |j| dst[pr + j * sr] = xs[ps + j * ss] }
          end
        end
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
        pos, stride = table.view.offset_of(id * width), table.view.run_stride
        width.times { |j| dst[i * width + j] = xs[pos + j * stride] }
      end
    end

    # Computes each output as a dot product over `k`, added in index order.
    #
    # 1. Finds where each batch matrix of both sources starts, once per batch.
    # 2. Walks rows, columns and `k` with the sources' strides directly, so the
    #    order of additions, and so every result bit, is the same as indexing
    #    each element through its view.
    private def matmul(type : T.class, node : Node, dst : Slice(T)) : Nil forall T
      a, b = node.srcs
      m, k = a.shape[-2], a.shape[-1]
      n = b.shape[-1]
      return if dst.empty?
      xs, ys = elements(T, a), elements(T, b)
      a_row, a_k = a.view.strides[-2], a.view.strides[-1]
      b_k, b_col = b.view.strides[-2], b.view.strides[-1]
      (dst.size // (m * n)).times do |batch|
        a_base = a.view.offset_of(batch * m * k)
        b_base = b.view.offset_of(batch * k * n)
        m.times do |row|
          n.times do |col|
            pa, pb = a_base + row * a_row, b_base + col * b_col
            acc = T.zero
            k.times { |step| acc += xs[pa + step * a_k] * ys[pb + step * b_k] }
            dst[(batch * m + row) * n + col] = acc
          end
        end
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
