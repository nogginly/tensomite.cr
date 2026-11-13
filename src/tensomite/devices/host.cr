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

    # For a subclass that holds its data elsewhere, such as in GPU memory, and
    # makes a host copy only when one is needed: *bytes* may be empty.
    def initialize(device : Device, bytesize : Int32, @bytes : Bytes)
      super(device, bytesize)
    end

    # Gives the memory back to the device's pool, unless it belongs to a file mapping.
    protected def give_back : Nil
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

    # The time each op takes, in milliseconds, and how many ran, by op name,
    # while set; `nil`, the default, times nothing. See
    # `bench/profile_decode_cpu.cr`.
    property op_times : Hash(String, {Int32, Float64})? = nil

    def execute(node : Node, output : Buffer) : Nil
      timed(node) do
        case node.dtype
        in .f32? then execute(Float32, node, output)
        in .i32? then execute(Int32, node, output)
        in .q8_0?, .q4_0?, .q4_1?
          raise DTypeError.new("no op computes #{node.dtype} values")
        end
      end
    end

    # Runs the block, adding its time to *node*'s op in `#op_times` when that
    # is set.
    protected def timed(node : Node, &) : Nil
      times = op_times
      return yield unless times
      start = Time.instant
      yield
      label = node.op.to_s
      count, total = times[label]? || {0, 0.0}
      times[label] = {count + 1, total + (Time.instant - start).total_milliseconds}
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
      in Op::Write        then copy_into(T, node.srcs.last, node.source_view || raise(Error.new("write needs a region")), dst)
      in Op::Gather       then gather(T, node, dst)
      in Op::MatMul       then matmul(T, node, dst)
      in Op::BlockMatMul  then block_matmul(node, Slice.new(dst.to_unsafe.as(Float32*), dst.size))
      in Op::GreaterEqual then binary(T, node, dst) { |x, y| x >= y ? T.new(1) : T.zero }
      in Op::Unview       then unview(T, node, dst)
      in Op::ScatterAdd   then scatter_add(T, node, dst)
      in Op::Rope         then rope(node, Slice.new(dst.to_unsafe.as(Float32*), dst.size))
      in Op::Softmax, Op::SoftmaxGrad, Op::LogSoftmax, Op::LogSoftmaxGrad, Op::LayerNorm, Op::LayerNormGrad, Op::RMSNorm
        rows(node, Slice.new(dst.to_unsafe.as(Float32*), dst.size))
      in Op::Dropout   then dropout(node, Slice.new(dst.to_unsafe.as(Float32*), dst.size))
      in Op::Attention then attention(node, Slice.new(dst.to_unsafe.as(Float32*), dst.size))
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
        HostDevice.check_run(pa, run, sa, xs.size) { describe(node) }
        HostDevice.check_run(pb, run, sb, ys.size) { describe(node) }
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
        HostDevice.check_run(pos, run, stride, xs.size) { describe(node) }
        input, output = xs.to_unsafe + pos, dst.to_unsafe + base
        if stride == 1
          run.times { |j| output[j] = yield input[j] }
        else
          run.times { |j| output[j] = yield input[j * stride] }
        end
      end
    end

    # Keeps logical element *i* of the source, times the scale, when its hash
    # clears the threshold, and writes 0 otherwise.
    private def dropout(node : Node, dst : Slice(Float32)) : Nil
      return if dst.empty?
      mask = node.dropout || raise Error.new("dropout without a mask")
      scale = node.scale || raise Error.new("dropout without a scale")
      src = node.srcs.first
      xs = elements(Float32, src)
      run = View.run_length(dst.size, {src.view})
      stride = src.view.run_stride
      (dst.size // run).times do |run_index|
        base = run_index * run
        pos = src.view.offset_of(base)
        HostDevice.check_run(pos, run, stride, xs.size) { describe(node) }
        input, output = xs.to_unsafe + pos, dst.to_unsafe + base
        run.times do |j|
          kept = Tensomite.dropout_hash(mask.seed, mask.stream, (base + j).to_u32) >= mask.threshold
          output[j] = kept ? input[j * stride] * scale : 0f32
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
        HostDevice.check_run(pv, run, sv, dst.size) { describe(node) }
        HostDevice.check_run(ps, run, ss, xs.size) { describe(node) }
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
    # - Softmax: `x · scale + mask`, when it has them; the maximum, folded from the
    #   first element; then `e = exp(x - max)` and their sum, folded from zero;
    #   then `e / sum`.
    # - SoftmaxGrad: the sum of `g · y`, folded from zero; then `y · (g - sum)`,
    #   times the scale when it has one.
    # - LogSoftmax: `shifted = x - max`; the sum of `exp(shifted)`; then
    #   `shifted - log(sum)`.
    # - LogSoftmaxGrad: the sum of `g`, folded from zero; then `g - exp(y) · sum`.
    # - LayerNormGrad: `mean = sum(x) / n`, `centred = x - mean`, `σ =
    #   sqrt(sum(centred²) / n + epsilon)`, `x̂ = centred / σ`; then either `x̂`,
    #   or, with `ĝ = g · weight`, `mean(ĝ)` and `mean(ĝ · x̂)` as sums over n,
    #   and `((ĝ - mean(ĝ)) - x̂ · mean(ĝ · x̂)) / σ`.
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
        when Op::Softmax
          if node.scale || node.srcs.size > 1
            scaled_softmax_row(node, xs, pos, stride, dst, base, n)
            next
          end
          HostDevice.check_run(pos, n, stride, xs.size) { describe(node) }
          row_in, row_out = xs.to_unsafe + pos, dst.to_unsafe + base
          top = row_in[0]
          (1...n).each { |j| top = (v = row_in[j * stride]) > top ? v : top }
          total = 0f32
          if stride == 1
            n.times { |j| row_out[j] = Maths.exp(row_in[j] - top) }
          else
            n.times { |j| row_out[j] = Maths.exp(row_in[j * stride] - top) }
          end
          n.times { |j| total += row_out[j] }
          n.times { |j| row_out[j] = row_out[j] / total }
        when Op::SoftmaxGrad, Op::LogSoftmaxGrad
          gradient_row(node, xs, pos, stride, dst, base, n)
        when Op::LayerNormGrad
          layernorm_grad_row(node, dst, base, n)
        when Op::RMSNorm
          rmsnorm_row(node, xs, pos, stride, dst, base, n)
        when Op::LogSoftmax
          HostDevice.check_run(pos, n, stride, xs.size) { describe(node) }
          row_in, row_out = xs.to_unsafe + pos, dst.to_unsafe + base
          top = row_in[0]
          (1...n).each { |j| top = (v = row_in[j * stride]) > top ? v : top }
          total = 0f32
          n.times { |j| row_out[j] = row_in[j * stride] - top }
          n.times { |j| total += Maths.exp(row_out[j]) }
          log_total = Maths.log(total)
          n.times { |j| row_out[j] = row_out[j] - log_total }
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

    # Writes the softmax of `x · scale + mask` for the row at *base*: each of
    # those operations rounded as its own op would round it, into the output
    # row, then the softmax of the row in place.
    private def scaled_softmax_row(node : Node, xs : Slice(Float32), pos : Int32, stride : Int32,
                                   dst : Slice(Float32), base : Int32, n : Int32) : Nil
      scale = node.scale
      mask = node.srcs[1]?
      ms = mask ? elements(Float32, mask) : xs
      pm, sm = mask ? {mask.view.offset_of(base), mask.view.run_stride} : {0, 0}
      row_out = dst.to_unsafe + base
      n.times do |j|
        x = xs[pos + j * stride]
        x = x * scale if scale
        x = x + ms[pm + j * sm] if mask
        row_out[j] = x
      end
      top = row_out[0]
      (1...n).each { |j| top = (v = row_out[j]) > top ? v : top }
      n.times { |j| row_out[j] = Maths.exp(row_out[j] - top) }
      total = 0f32
      n.times { |j| total += row_out[j] }
      n.times { |j| row_out[j] = row_out[j] / total }
    end

    # Writes `x / √(Σx²/n + ε) · w` for the row at *base*: the sum of squares
    # folded from zero, then each operation rounded as the composite's op would.
    private def rmsnorm_row(node : Node, xs : Slice(Float32), pos : Int32, stride : Int32,
                            dst : Slice(Float32), base : Int32, n : Int32) : Nil
      weight = node.srcs[1]
      ws, pw, sw = elements(Float32, weight), weight.view.offset_of(base), weight.view.run_stride
      epsilon = node.epsilon || raise Error.new("rmsnorm needs an epsilon")
      squares = 0f32
      n.times { |j| squares += xs[pos + j * stride] * xs[pos + j * stride] }
      root = Math.sqrt(squares / n.to_f32 + epsilon)
      row_out = dst.to_unsafe + base
      n.times { |j| row_out[j] = xs[pos + j * stride] / root * ws[pw + j * sw] }
    end

    # Writes attention for each query row: its scores `q · k · scale + mask`
    # against every position, their softmax, and the values weighted by it,
    # each sum in index order. The host makes no `Attention` nodes itself (see
    # `Tensor.attention`); this computes one a GPU leaves to it.
    private def attention(node : Node, dst : Slice(Float32)) : Nil
      q, k, v = node.srcs
      mask = node.srcs[3]?
      qs, ks, vs = elements(Float32, q), elements(Float32, k), elements(Float32, v)
      ms = mask ? elements(Float32, mask) : qs
      width, positions, out_width = q.shape[-1], k.shape[-2], v.shape[-1]
      rows = q.shape[-2]
      scale = node.scale || raise Error.new("attention needs a scale")
      scores = Slice(Float32).new(positions)
      (q.size // width).times do |item|
        head = item // rows
        top = Float32::MIN
        positions.times do |position|
          dot = 0f32
          width.times do |col|
            dot += qs[q.view.offset_of(item * width + col)] * ks[k.view.offset_of((head * positions + position) * width + col)]
          end
          x = dot * scale
          x = x + ms[mask.view.offset_of(item * positions + position)] if mask
          scores[position] = x
          top = x if position == 0 || x > top
        end
        total = 0f32
        positions.times { |position| total += (scores[position] = Maths.exp(scores[position] - top)) }
        out_width.times do |col|
          acc = 0f32
          positions.times do |position|
            acc += scores[position] / total * vs[v.view.offset_of((head * positions + position) * out_width + col)]
          end
          dst[item * out_width + col] = acc
        end
      end
    end

    # Writes `x · cos + rotate_half(x) · sin` for each element, where
    # `rotate_half(x)` at a column of the first half is the element half a row
    # on, times -1, and at a column of the second half the element half a row
    # back, as the composite's ops compute it. Each row's starts are found
    # once, and its elements read along their views' last strides.
    private def rope(node : Node, dst : Slice(Float32)) : Nil
      x, cos, sin = node.srcs
      xs, cs, ss = elements(Float32, x), elements(Float32, cos), elements(Float32, sin)
      n = node.shape.last
      half = n // 2
      sx, sc, ssn = x.view.run_stride, cos.view.run_stride, sin.view.run_stride
      (dst.size // n).times do |row|
        base = row * n
        px, pc, ps = x.view.offset_of(base), cos.view.offset_of(base), sin.view.offset_of(base)
        HostDevice.check_run(px, n, sx, xs.size) { describe(node) }
        HostDevice.check_run(pc, n, sc, cs.size) { describe(node) }
        HostDevice.check_run(ps, n, ssn, ss.size) { describe(node) }
        n.times do |col|
          partner = xs.unsafe_fetch(px + (col < half ? col + half : col - half) * sx)
          rotated = col < half ? partner * -1f32 : partner
          dst[base + col] = xs.unsafe_fetch(px + col * sx) * cs.unsafe_fetch(pc + col * sc) + rotated * ss.unsafe_fetch(ps + col * ssn)
        end
      end
    end

    # Writes the row at *base* of a softmax's or log-softmax's gradient.
    private def gradient_row(node : Node, gs : Slice(Float32), pg : Int32, sg : Int32,
                             dst : Slice(Float32), base : Int32, n : Int32) : Nil
      if node.op == Op::SoftmaxGrad
        softmax_grad_row(node, gs, pg, sg, dst, base, n)
      else
        log_softmax_grad_row(node, gs, pg, sg, dst, base, n)
      end
    end

    # Writes `y · (g - Σ g·y)`, times the scale when the node has one, for the
    # row at *base*, from the gradient *gs* and the softmax's output (source 1).
    private def softmax_grad_row(node : Node, gs : Slice(Float32), pg : Int32, sg : Int32,
                                 dst : Slice(Float32), base : Int32, n : Int32) : Nil
      y = node.srcs[1]
      ys, py, sy = elements(Float32, y), y.view.offset_of(base), y.view.run_stride
      total = 0f32
      n.times { |j| total += gs[pg + j * sg] * ys[py + j * sy] }
      scale = node.scale
      row_out = dst.to_unsafe + base
      n.times do |j|
        value = ys[py + j * sy] * (gs[pg + j * sg] - total)
        row_out[j] = scale ? value * scale : value
      end
    end

    # Writes `g - exp(y) · Σ g` for the row at *base*, from the gradient *gs*
    # and the log-softmax's output (source 1).
    private def log_softmax_grad_row(node : Node, gs : Slice(Float32), pg : Int32, sg : Int32,
                                     dst : Slice(Float32), base : Int32, n : Int32) : Nil
      y = node.srcs[1]
      ys, py, sy = elements(Float32, y), y.view.offset_of(base), y.view.run_stride
      total = 0f32
      n.times { |j| total += gs[pg + j * sg] }
      row_out = dst.to_unsafe + base
      n.times { |j| row_out[j] = gs[pg + j * sg] - Maths.exp(ys[py + j * sy]) * total }
    end

    # Writes the row at *base* of a `LayerNormGrad`: the normalised input, or,
    # given g and the weight, the input's gradient, each operation rounded as
    # the composite's own op would round it.
    private def layernorm_grad_row(node : Node, dst : Slice(Float32), base : Int32, n : Int32) : Nil
      x = node.srcs.size == 1 ? node.srcs[0] : node.srcs[1]
      xs, px, sx = elements(Float32, x), x.view.offset_of(base), x.view.run_stride
      epsilon = node.epsilon || raise Error.new("layernorm needs an epsilon")
      count = n.to_f32
      total = 0f32
      n.times { |j| total += xs[px + j * sx] }
      mean = total / count
      squares = 0f32
      n.times do |j|
        centred = xs[px + j * sx] - mean
        squares += centred * centred
      end
      std = Math.sqrt(squares / count + epsilon)
      row_out = dst.to_unsafe + base
      if node.srcs.size == 1
        n.times { |j| row_out[j] = (xs[px + j * sx] - mean) / std }
        return
      end
      g, weight = node.srcs[0], node.srcs[2]
      gs, pg, sg = elements(Float32, g), g.view.offset_of(base), g.view.run_stride
      ws, pw, sw = elements(Float32, weight), weight.view.offset_of(base), weight.view.run_stride
      # ĝ, then x̂, into the output row, so that each is computed once.
      n.times { |j| row_out[j] = gs[pg + j * sg] * ws[pw + j * sw] }
      scaled_total = 0f32
      n.times { |j| scaled_total += row_out[j] }
      scaled_mean = scaled_total / count
      product_total = 0f32
      n.times { |j| product_total += row_out[j] * ((xs[px + j * sx] - mean) / std) }
      product_mean = product_total / count
      n.times do |j|
        normalised = (xs[px + j * sx] - mean) / std
        row_out[j] = (row_out[j] - scaled_mean - normalised * product_mean) / std
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
        stop = start + src.shape[axis]
        copy_into(T, src, out_view.slice(axis, start, stop), dst)
        start = stop
      end
    end

    # Copies *src* into the positions *region* gives in *dst*, leaving the rest.
    private def copy_into(type : T.class, src : Tensor, region : View, dst : Slice(T)) : Nil forall T
      return if src.size == 0
      xs = elements(T, src)
      run = View.run_length(src.size, {region, src.view})
      sr, ss = region.run_stride, src.view.run_stride
      (src.size // run).times do |run_index|
        base = run_index * run
        pr, ps = region.offset_of(base), src.view.offset_of(base)
        run.times { |j| dst[pr + j * sr] = xs[ps + j * ss] }
      end
    end

    # Copies row `ids[i]` of the table into output row `i`.
    private def gather(type : T.class, node : Node, dst : Slice(T)) : Nil forall T
      table, ids = node.srcs
      rows, width = table.shape
      return block_gather(node, Slice.new(dst.to_unsafe.as(Float32*), dst.size)) if table.dtype.block?
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

    # Gathers rows of a block-format table, dequantising each (see `Quant.gather`).
    private def block_gather(node : Node, dst : Slice(Float32)) : Nil
      table, ids = node.srcs
      rows, width = table.shape
      idx = elements(Int32, ids)
      chosen = Array(Int32).new(ids.size) do |i|
        id = idx[ids.view.offset_of(i)]
        raise ExecutionError.new("index #{id} out of range for #{rows} rows") unless 0 <= id < rows
        id + table.view.offset // width
      end
      Quant.gather(table.dtype, own(table.node.buffer!).bytes, width, chosen, dst)
    end

    # Computes `x · wᵀ` for a block-format weight (see `Quant.matmul`), then adds
    # the bias, if any.
    private def block_matmul(node : Node, dst : Slice(Float32)) : Nil
      x, w = node.srcs[0], node.srcs[1]
      n, k = w.shape
      rows = x.size // k
      values = activations(node, x)
      weights = own(w.node.buffer!).bytes
      start = w.dtype.bytes_for(w.view.offset)
      Quant.matmul(w.dtype, values, weights[start, w.dtype.bytes_for(w.size)], rows, n, k, dst)
      add_bias(Float32, node, dst)
    end

    # Returns *x*'s values in row-major order: its buffer in place when the
    # view is dense, or a copy walked in runs (see `View.run_length`).
    private def activations(node : Node, x : Tensor) : Slice(Float32)
      xs = elements(Float32, x)
      view = x.view
      if view.dense?
        HostDevice.check_run(view.offset, x.size, 1, xs.size) { describe(node) }
        return xs[view.offset, x.size]
      end
      run, stride = View.run_length(x.size, {view}), view.run_stride
      values = Slice(Float32).new(x.size)
      dst = values.to_unsafe
      (x.size // run).times do |index|
        start = view.offset_of(index * run)
        HostDevice.check_run(start, run, stride, xs.size) { describe(node) }
        src = xs.to_unsafe + start
        run.times { |j| dst[index * run + j] = src[j * stride] }
      end
      values
    end

    # Computes each output as a dot product over `k`, added in index order.
    #
    # 1. Finds where each batch matrix of both sources starts, once per batch.
    # 2. Walks rows, columns and `k` with the sources' strides directly, so the
    #    order of additions, and so every result bit, is the same as indexing
    #    each element through its view.
    private def matmul(type : T.class, node : Node, dst : Slice(T)) : Nil forall T
      a, b = node.srcs[0], node.srcs[1]
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
      add_bias(T, node, dst)
    end

    # Adds a `MatMul`'s bias, its third source if it has one, to each row of
    # *dst*, as a separate `Add` would.
    protected def add_bias(type : T.class, node : Node, dst : Slice(T)) : Nil forall T
      return unless bias = node.srcs[2]?
      n = bias.size
      return if n == 0
      cs = elements(T, bias)[0, n].to_unsafe
      (dst.size // n).times do |row|
        out = dst.to_unsafe + row * n
        n.times { |j| out[j] = out[j] + cs[j] }
      end
    end

    # Raises `IndexError`, described by the block, unless every element of a run of
    # *count* from *start*, *stride* apart, lies within a buffer of *size* elements.
    # Kernels check each run once, then read it through pointers.
    def self.check_run(start : Int32, count : Int32, stride : Int32, size : Int32, & : -> String) : Nil
      return if 0 <= start && 0 <= start + (count - 1) * stride < size
      raise IndexError.new("#{yield}: a run of #{count} from #{start}, stride #{stride}, leaves a buffer of #{size}")
    end

    private def describe(node : Node) : String
      "#{node.op} #{node.shape} from #{node.srcs.map { |src| "#{src.shape} #{src.view.inspect}" }.join(", ")}"
    end

    private def elements(type : T.class, tensor : Tensor) : Slice(T) forall T
      elements(T, tensor.node.buffer!)
    end

    private def elements(type : T.class, buffer : Buffer) : Slice(T) forall T
      own(buffer).bytes.unsafe_slice_of(T)
    end
  end
end
