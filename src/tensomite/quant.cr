module Tensomite
  # Quantises and dequantises rows in ggml's block formats on the host, giving
  # the bytes and values of ggml's reference code (`quantize_row_*_ref` and
  # `dequantize_row_*`).
  #
  # Each block holds 32 consecutive values:
  #
  # - `Q8_0`: a scale `d` (`F16`), then 32 signed bytes; `x = q · d`.
  # - `Q4_0`: `d` (`F16`), then 16 bytes, each holding element `j` in its low
  #   nibble and element `j + 16` in its high one; `x = (q - 8) · d`.
  # - `Q4_1`: `d` and a minimum `m` (both `F16`), then nibbles as in `Q4_0`;
  #   `x = q · d + m`.
  # How a GPU divides the work of a quantised matmul (`Tensor#linear` with a
  # block-format weight). Every mode but `Unordered` gives the host's bits;
  # they differ only in speed.
  enum BlockMatMulMode
    # Lanes when the product has fewer than `.lanes_below` outputs, which leave
    # one thread per output too few to fill the GPU, and threads otherwise.
    Auto
    # One thread per output, adding its blocks in order.
    Thread
    # 32 lanes per output: each lane takes whole blocks, and the lanes' sums
    # are added in block order.
    Lanes
    # For measurement only: as `Lanes`, but each lane keeps one running sum
    # over its blocks and the lanes' sums are added in a tree, so the bits can
    # differ from the host's. It shows what adding in block order costs.
    Unordered

    # Returns the number of outputs below which `Auto` uses lanes. On the GB10,
    # lanes decode SmolLM2's 49,152-word head faster than threads too.
    def self.lanes_below : Int32
      65_536
    end

    # Returns the mode `TENSOMITE_BLOCK_MATMUL` names (`auto`, `thread`,
    # `lanes` or `unordered`), or `Auto` without it, so that the modes can be timed against
    # each other without rebuilding.
    def self.from_env : BlockMatMulMode
      ENV["TENSOMITE_BLOCK_MATMUL"]?.try { |name| parse?(name) } || Auto
    end

    # Returns the outputs each warp or SIMD group computes when a product runs
    # on lanes: *chosen*, the device's own choice, or the count
    # `TENSOMITE_LANE_OUTPUTS` (1 to 8) sets, so that the counts can be timed
    # against each other without rebuilding.
    def self.lane_outputs(chosen : Int32) : Int32
      ENV["TENSOMITE_LANE_OUTPUTS"]?.try(&.to_i?).try(&.clamp(1, 8)) || chosen
    end

    # Returns true when a product of *outputs* outputs runs on lanes, in block
    # order or not.
    def lanes?(outputs : Int32) : Bool
      lanes? || unordered? || (auto? && outputs < BlockMatMulMode.lanes_below)
    end
  end

  module Quant
    extend self

    # Returns *values*, a whole number of blocks, quantised to *dtype*.
    def quantize(dtype : DType, values : Slice(Float32)) : Bytes
      bytes = Bytes.new(dtype.bytes_for(values.size))
      (values.size // DType.block_size).times do |block|
        source = values[block * DType.block_size, DType.block_size]
        target = bytes[block * dtype.block_bytes, dtype.block_bytes]
        case dtype
        when .q8_0? then quantize_q8_0(source, target)
        when .q4_0? then quantize_q4_0(source, target)
        when .q4_1? then quantize_q4_1(source, target)
        else             raise DTypeError.new("#{dtype} is not a block format")
        end
      end
      bytes
    end

    # Writes the values of *bytes*, blocks of *dtype*, into *values*.
    def dequantize(dtype : DType, bytes : Bytes, values : Slice(Float32)) : Nil
      raise ArgumentError.new("#{bytes.size} bytes do not hold #{values.size} #{dtype} values") unless bytes.size == dtype.bytes_for(values.size)
      (values.size // DType.block_size).times do |block|
        source = bytes[block * dtype.block_bytes, dtype.block_bytes]
        target = values[block * DType.block_size, DType.block_size]
        case dtype
        when .q8_0? then dequantize_q8_0(source, target)
        when .q4_0? then dequantize_q4_0(source, target)
        when .q4_1? then dequantize_q4_1(source, target)
        else             raise DTypeError.new("#{dtype} is not a block format")
        end
      end
    end

    # Writes `x · wᵀ` into *dst*, `[rows, n]`, for *x*, `rows` rows of *k*
    # values, and *w*, `n` rows of *k* values in *dtype*, as ggml's reference
    # dot products compute it: the naive oracle `#matmul` is checked against,
    # and gives the bits of.
    #
    # 1. Quantises each row of *x* once, block by block: to `Q8_0` for `Q8_0`
    #    and `Q4_0` weights, and to `Q8_1` (which also keeps `d · Σq` in `F16`)
    #    for `Q4_1`.
    # 2. For each output, sums the blocks in order: the integer dot product of
    #    the block's quantised values, times the weight's and the activation's
    #    scales, plus, for `Q4_1`, the weight's minimum times the activation's
    #    `d · Σq`.
    #
    # Each output depends only on its own row and column, so the bits do not
    # change with the number of rows.
    def matmul_reference(dtype : DType, x : Slice(Float32), w : Bytes, rows : Int32, n : Int32, k : Int32, dst : Slice(Float32)) : Nil
      blocks = k // DType.block_size
      row_bytes = dtype.bytes_for(k)
      scales = Slice(Float32).new(blocks)
      sums = Slice(Float32).new(blocks)
      qs = Slice(Int8).new(k)
      raise ArgumentError.new("#{w.size} bytes do not hold #{n} rows of #{k} #{dtype} values") unless w.size == n * row_bytes
      raise ArgumentError.new("#{x.size} values are not #{rows} rows of #{k}") unless x.size == rows * k && dst.size == rows * n
      rows.times do |row|
        quantize_activations(x[row * k, k], scales, sums, qs, with_sums: dtype.q4_1?)
        n.times do |col|
          weight = w[col * row_bytes, row_bytes]
          dst[row * n + col] = case dtype
                               when .q8_0? then dot_q8_0(weight, scales, qs, blocks)
                               when .q4_0? then dot_q4_0(weight, scales, qs, blocks)
                               when .q4_1? then dot_q4_1(weight, scales, sums, qs, blocks)
                               else             raise DTypeError.new("#{dtype} is not a block format")
                               end
        end
      end
    end

    # Writes `x · wᵀ` into *dst* as `#matmul_reference` does, with its bits,
    # faster:
    #
    # 1. Quantises each row of *x* once, as the reference does, and also keeps
    #    each block's integer sum of quantised values.
    # 2. For each pair of outputs, sums the blocks in order, sharing each load
    #    of the activations. A block's integer dot product is taken in
    #    wrapping arithmetic on signed bytes, which LLVM vectorises, as `sdot`
    #    where the target has it: nibbles are unpacked less 8, so for `Q4_1`
    #    it is `Σ(q - 8)·y + 8·Σy`, the same integer as the reference's
    #    `Σq·y`. The floating-point expressions after it are the reference's.
    def matmul(dtype : DType, x : Slice(Float32), w : Bytes, rows : Int32, n : Int32, k : Int32, dst : Slice(Float32)) : Nil
      blocks = k // DType.block_size
      row_bytes = dtype.bytes_for(k)
      scales = Slice(Float32).new(blocks)
      sums = Slice(Float32).new(blocks)
      totals = Slice(Int32).new(blocks)
      qs = Slice(Int8).new(k)
      raise ArgumentError.new("#{w.size} bytes do not hold #{n} rows of #{k} #{dtype} values") unless w.size == n * row_bytes
      raise ArgumentError.new("#{x.size} values are not #{rows} rows of #{k}") unless x.size == rows * k && dst.size == rows * n
      rows.times do |row|
        fast_quantize(x[row * k, k].to_unsafe, scales, sums, totals, qs, with_sums: dtype.q4_1?)
        out = dst[row * n, n]
        Quant.in_parallel(n, n * row_bytes) do |first, stop|
          first.step(to: stop - 1, by: 2) do |col|
            w0 = w.to_unsafe + col * row_bytes
            w1 = col + 1 < stop ? w0 + row_bytes : w0
            a, b = case dtype
                   when .q8_0? then fast_q8_0(w0, w1, scales, qs, blocks)
                   when .q4_0? then fast_q4_0(w0, w1, scales, qs, blocks)
                   when .q4_1? then fast_q4_1(w0, w1, scales, sums, totals, qs, blocks)
                   else             raise DTypeError.new("#{dtype} is not a block format")
                   end
            out[col] = a
            out[col + 1] = b if col + 1 < stop
          end
        end
      end
    end

    # The threads quantised products run on: `TENSOMITE_THREADS`, or the
    # machine's CPU count when it is unset or not positive. Setting it after the first product has shared its
    # work changes how many spans products are cut into, not the pool's size.
    class_property threads : Int32 = ENV["TENSOMITE_THREADS"]?.try(&.to_i?).try { |count| count.clamp(1, 1024) if count > 0 } || System.cpu_count.to_i.clamp(1, 1024)

    # The fewest weight bytes worth a span of their own.
    private PARALLEL_BYTES = 64 * 1024

    # Spans per thread, so that faster cores take more of them.
    private SPANS_PER_THREAD = 4

    @@pool : SpinPool? = nil

    # Runs the block on spans `first...stop` that cover `0...count` in order:
    # one span on the calling thread when the work, *bytes* of weights, is too
    # small to share, or up to `SPANS_PER_THREAD` spans per thread, taken in
    # turn by the calling thread and the pool's, otherwise; returns when all
    # have finished. Each index falls in exactly one span, so the results do
    # not depend on the number of threads.
    protected def self.in_parallel(count : Int32, bytes : Int32, &block : Int32, Int32 ->) : Nil
      spans = Math.min(threads * SPANS_PER_THREAD, Math.min(count, bytes // PARALLEL_BYTES))
      return block.call(0, count) if spans <= 1 || threads <= 1
      pool = @@pool ||= SpinPool.new(threads - 1)
      pool.run(spans, count, block)
    end

    # Worker threads that wait for work by spinning, so that a product's spans
    # start within a microsecond rather than when a sleeping thread wakes.
    # Each runs in its own isolated execution context; with the calling
    # thread, they are as many as `Quant.threads`, so that no thread spinning
    # waits for a core. A worker that has seen no work for a while naps
    # between checks, each nap twice the last, up to `LONGEST_NAP`.
    private class SpinPool
      # Checks a worker spins through before it starts to nap.
      SPINS = 1_000_000

      FIRST_NAP   = 100.microseconds
      LONGEST_NAP = 10.milliseconds

      # The current product's generation in the high 32 bits and its next
      # unclaimed span in the low 32: a span is claimed by compare-and-set,
      # so a claim made for one product fails once the next has started.
      @claim = Atomic(Int64).new(0)
      @done = Atomic(Int32).new(0)
      @spans = 0
      @count = 0
      @job : Proc(Int32, Int32, Nil)? = nil

      def initialize(workers : Int32)
        workers.times do |index|
          Fiber::ExecutionContext::Isolated.new("tensomite-#{index}") { work }
        end
      end

      # Runs *job* on *spans* spans of `0...count`: publishes it, takes spans
      # alongside the workers, then waits until every span has finished.
      #
      # The job, span count and counters are written before the claim moves
      # to the new generation, so a worker that sees the generation sees them
      # too.
      def run(spans : Int32, count : Int32, job : Proc(Int32, Int32, Nil)) : Nil
        @job, @spans, @count = job, spans, count
        @done.set(0)
        generation = (@claim.get >> 32) &+ 1
        @claim.set(generation << 32)
        help(generation)
        until @done.get == spans
          Intrinsics.pause
        end
      end

      # Waits for each new generation, spinning, then napping, and takes its
      # spans.
      private def work : Nil
        seen = 0_i64
        loop do
          idle = 0
          nap = FIRST_NAP
          while (current = @claim.get >> 32) == seen
            if idle < SPINS
              idle += 1
              Intrinsics.pause
            else
              sleep nap
              nap = Math.min(nap * 2, LONGEST_NAP)
            end
          end
          seen = current
          help(current)
        end
      end

      # Takes spans of product *generation* until none are left, running the
      # job on each.
      #
      # A claim succeeds only while the product is unfinished, so the next
      # product has not been published, and the span count, count and job
      # read before it are this product's.
      private def help(generation : Int64) : Nil
        loop do
          claim = @claim.get
          return unless claim >> 32 == generation
          span = (claim & 0xffff_ffff).to_i32
          spans, count, job = @spans, @count, @job
          return unless span < spans
          next unless @claim.compare_and_set(claim, claim + 1)[1]
          job.try &.call(count * span // spans, count * (span + 1) // spans)
          @done.add(1)
        end
      end
    end

    # Returns *value* rounded to the nearest integer, ties away from zero, as
    # `round(:ties_away)` does for every value within `Int32`'s range, without
    # a call to `roundf`, so that loops of it vectorise: truncates, then steps
    # away from zero when the fraction dropped is at least a half.
    @[AlwaysInline]
    private def round_away(value : Float32) : Int32
      whole = value.to_i32!
      fraction = value - whole.to_f32
      whole &+ (fraction >= 0.5_f32 ? 1 : 0) &- (fraction <= -0.5_f32 ? 1 : 0)
    end

    # Quantises *x* as `#quantize_activations` does, with its bits, through
    # pointers, and also writes each block's integer sum of quantised values
    # to *totals*. The caller has checked that *x* holds `qs.size` values.
    private def fast_quantize(x : Float32*, scales : Slice(Float32), sums : Slice(Float32), totals : Slice(Int32), qs : Slice(Int8), with_sums : Bool) : Nil
      sp, mp, tp, qp = scales.to_unsafe, sums.to_unsafe, totals.to_unsafe, qs.to_unsafe
      scales.size.times do |block|
        values = x + block * 32
        top = values[0].abs
        1.upto(31) { |j| top = values[j].abs if values[j].abs > top }
        d = top / 127.0_f32
        id = d == 0 ? 0.0_f32 : 1.0_f32 / d
        qb = qp + block &* 32
        32.times { |j| qb[j] = round_away(values[j] * id).to_i8! }
        total = 0
        32.times { |j| total &+= qb[j].to_i32 }
        tp[block] = total
        sp[block] = F16.new(d).to_f32
        mp[block] = F16.new(total.to_f32 * d).to_f32 if with_sums
      end
    end

    @[AlwaysInline]
    private def fast_half(at : UInt8*) : Float32
      F16.from_bits(at[0].to_u16 | at[1].to_u16 << 8).to_f32
    end

    # The products of two weight rows, at *w0* and *w1*, with one row of
    # quantised activations, each output's blocks summed in order as the
    # reference sums them. The two rows share each load of the activations.
    private def fast_q8_0(w0 : UInt8*, w1 : UInt8*, scales : Slice(Float32), qs : Slice(Int8), blocks : Int32) : {Float32, Float32}
      sum0 = sum1 = 0.0_f32
      sp, qp = scales.to_unsafe, qs.to_unsafe
      blocks.times do |block|
        base0, base1 = w0 + block &* 34, w1 + block &* 34
        v0, v1 = (base0 + 2).as(Int8*), (base1 + 2).as(Int8*)
        qb = qp + block &* 32
        total0 = total1 = 0
        32.times do |j|
          y = qb[j].to_i32
          total0 &+= v0[j].to_i32 &* y
          total1 &+= v1[j].to_i32 &* y
        end
        sum0 += total0.to_f32 * (fast_half(base0) * sp[block])
        sum1 += total1.to_f32 * (fast_half(base1) * sp[block])
      end
      {sum0, sum1}
    end

    # Returns the integer dot products of two blocks' 32 nibbles less 8, at
    # *p0* and *p1*, low halves first, with the same 32 quantised
    # activations. Both sides are signed bytes, so LLVM turns each loop into
    # `sdot` where the target has it.
    @[AlwaysInline]
    private def nibble_dot(p0 : UInt8*, p1 : UInt8*, qb : Int8*) : {Int32, Int32}
      total0 = total1 = 0
      16.times do |j|
        y = qb[j].to_i32
        total0 &+= ((p0[j] & 0x0f).to_i8! &- 8).to_i32 &* y
        total1 &+= ((p1[j] & 0x0f).to_i8! &- 8).to_i32 &* y
      end
      16.times do |j|
        y = qb[j + 16].to_i32
        total0 &+= ((p0[j] >> 4).to_i8! &- 8).to_i32 &* y
        total1 &+= ((p1[j] >> 4).to_i8! &- 8).to_i32 &* y
      end
      {total0, total1}
    end

    private def fast_q4_0(w0 : UInt8*, w1 : UInt8*, scales : Slice(Float32), qs : Slice(Int8), blocks : Int32) : {Float32, Float32}
      sum0 = sum1 = 0.0_f32
      sp, qp = scales.to_unsafe, qs.to_unsafe
      blocks.times do |block|
        base0, base1 = w0 + block &* 18, w1 + block &* 18
        qb = qp + block &* 32
        total0, total1 = nibble_dot(base0 + 2, base1 + 2, qb)
        sum0 += total0.to_f32 * fast_half(base0) * sp[block]
        sum1 += total1.to_f32 * fast_half(base1) * sp[block]
      end
      {sum0, sum1}
    end

    private def fast_q4_1(w0 : UInt8*, w1 : UInt8*, scales : Slice(Float32), sums : Slice(Float32), totals : Slice(Int32), qs : Slice(Int8), blocks : Int32) : {Float32, Float32}
      sum0 = sum1 = 0.0_f32
      sp, mp, tp, qp = scales.to_unsafe, sums.to_unsafe, totals.to_unsafe, qs.to_unsafe
      blocks.times do |block|
        base0, base1 = w0 + block &* 20, w1 + block &* 20
        qb = qp + block &* 32
        dot0, dot1 = nibble_dot(base0 + 4, base1 + 4, qb)
        shift = 8 &* tp[block]
        sum0 += (fast_half(base0) * sp[block]) * (dot0 &+ shift).to_f32 + fast_half(base0 + 2) * mp[block]
        sum1 += (fast_half(base1) * sp[block]) * (dot1 &+ shift).to_f32 + fast_half(base1 + 2) * mp[block]
      end
      {sum0, sum1}
    end

    # Writes rows *ids* of *table*, rows of *width* values in *dtype*, into
    # *dst* as `F32`.
    def gather(dtype : DType, table : Bytes, width : Int32, ids : Enumerable(Int32), dst : Slice(Float32)) : Nil
      row_bytes = dtype.bytes_for(width)
      ids.each_with_index do |id, i|
        dequantize(dtype, table[id * row_bytes, row_bytes], dst[i * width, width])
      end
    end

    # Quantises *x* to 8 bits per block, as ggml's `quantize_row_q8_0` and
    # `quantize_row_q8_1` do: each block's scale is its largest magnitude over
    # 127, kept in `F16`, and each value is rounded halves away from zero. With
    # *with_sums*, each block's `d · Σq`, also in `F16`, goes into *sums*.
    private def quantize_activations(x : Slice(Float32), scales : Slice(Float32), sums : Slice(Float32), qs : Slice(Int8), with_sums : Bool) : Nil
      scales.size.times do |block|
        values = x[block * 32, 32]
        d = values.max_of(&.abs) / 127.0_f32
        id = d == 0 ? 0.0_f32 : 1.0_f32 / d
        total = 0
        32.times do |j|
          q = (values[j] * id).round(:ties_away).to_i8!
          qs[block * 32 + j] = q
          total += q
        end
        scales[block] = F16.new(d).to_f32
        sums[block] = F16.new(total.to_f32 * d).to_f32 if with_sums
      end
    end

    # The three dot products read through pointers: `matmul` has checked that
    # the weight row holds `blocks` blocks and the activations as many.
    private def dot_q8_0(w : Bytes, scales : Slice(Float32), qs : Slice(Int8), blocks : Int32) : Float32
      sum = 0.0_f32
      wp, sp, qp = w.to_unsafe, scales.to_unsafe, qs.to_unsafe
      blocks.times do |block|
        base = wp + block * 34
        qb = qp + block &* 32
        total = 0
        32.times { |j| total += base[2 + j].to_i8!.to_i32 * qb[j] }
        sum += total.to_f32 * (half(base) * sp[block])
      end
      sum
    end

    private def dot_q4_0(w : Bytes, scales : Slice(Float32), qs : Slice(Int8), blocks : Int32) : Float32
      sum = 0.0_f32
      wp, sp, qp = w.to_unsafe, scales.to_unsafe, qs.to_unsafe
      blocks.times do |block|
        base = wp + block * 18
        qb = qp + block &* 32
        low = high = 0
        16.times do |j|
          packed = base[2 + j]
          low += ((packed & 0x0f).to_i32 - 8) * qb[j]
          high += ((packed >> 4).to_i32 - 8) * qb[j + 16]
        end
        sum += (low + high).to_f32 * half(base) * sp[block]
      end
      sum
    end

    private def dot_q4_1(w : Bytes, scales : Slice(Float32), sums : Slice(Float32), qs : Slice(Int8), blocks : Int32) : Float32
      sum = 0.0_f32
      wp, sp, mp, qp = w.to_unsafe, scales.to_unsafe, sums.to_unsafe, qs.to_unsafe
      blocks.times do |block|
        base = wp + block * 20
        qb = qp + block &* 32
        low = high = 0
        16.times do |j|
          packed = base[4 + j]
          low += (packed & 0x0f).to_i32 * qb[j]
          high += (packed >> 4).to_i32 * qb[j + 16]
        end
        sum += (half(base) * sp[block]) * (low + high).to_f32 + half(base + 2) * mp[block]
      end
      sum
    end

    private def half(at : UInt8*) : Float32
      F16.from_bits(at[0].to_u16 | at[1].to_u16 << 8).to_f32
    end

    # The scale is the largest magnitude over 127; each value is rounded to
    # the nearest integer of the scale, halves away from zero.
    private def quantize_q8_0(x : Slice(Float32), target : Bytes) : Nil
      amax = x.max_of(&.abs)
      d = amax / 127.0_f32
      id = d == 0 ? 0.0_f32 : 1.0_f32 / d
      write_half(target, 0, d)
      32.times { |j| target[2 + j] = (x[j] * id).round(:ties_away).to_i8!.to_u8! }
    end

    # The scale is the value of largest magnitude, sign kept, over -8, so that
    # it maps to -8; each value maps to `trunc(x / d + 8.5)`, at most 15.
    private def quantize_q4_0(x : Slice(Float32), target : Bytes) : Nil
      max = x[0]
      x.each { |value| max = value if value.abs > max.abs }
      d = max / -8.0_f32
      id = d == 0 ? 0.0_f32 : 1.0_f32 / d
      write_half(target, 0, d)
      16.times do |j|
        low = Math.min(15, (x[j] * id + 8.5_f32).to_i8!)
        high = Math.min(15, (x[j + 16] * id + 8.5_f32).to_i8!)
        target[2 + j] = (low.to_u8! | high.to_u8! << 4)
      end
    end

    # The scale spans the block's range in 15 steps from its minimum; each
    # value maps to `trunc((x - m) / d + 0.5)`, at most 15.
    private def quantize_q4_1(x : Slice(Float32), target : Bytes) : Nil
      min, max = x.min, x.max
      d = (max - min) / 15.0_f32
      id = d == 0 ? 0.0_f32 : 1.0_f32 / d
      write_half(target, 0, d)
      write_half(target, 2, min)
      16.times do |j|
        low = Math.min(15, ((x[j] - min) * id + 0.5_f32).to_i8!)
        high = Math.min(15, ((x[j + 16] - min) * id + 0.5_f32).to_i8!)
        target[4 + j] = (low.to_u8! | high.to_u8! << 4)
      end
    end

    private def dequantize_q8_0(block : Bytes, x : Slice(Float32)) : Nil
      d = read_half(block, 0)
      32.times { |j| x[j] = block[2 + j].to_i8!.to_f32 * d }
    end

    private def dequantize_q4_0(block : Bytes, x : Slice(Float32)) : Nil
      d = read_half(block, 0)
      16.times do |j|
        x[j] = ((block[2 + j] & 0x0f).to_i32 - 8).to_f32 * d
        x[j + 16] = ((block[2 + j] >> 4).to_i32 - 8).to_f32 * d
      end
    end

    private def dequantize_q4_1(block : Bytes, x : Slice(Float32)) : Nil
      d, m = read_half(block, 0), read_half(block, 2)
      16.times do |j|
        x[j] = d * (block[4 + j] & 0x0f).to_f32 + m
        x[j + 16] = d * (block[4 + j] >> 4).to_f32 + m
      end
    end

    private def write_half(target : Bytes, at : Int32, value : Float32) : Nil
      bits = F16.new(value).bits
      target[at] = (bits & 0xff).to_u8
      target[at + 1] = (bits >> 8).to_u8
    end

    private def read_half(block : Bytes, at : Int32) : Float32
      F16.from_bits(block[at].to_u16 | block[at + 1].to_u16 << 8).to_f32
    end
  end
end
