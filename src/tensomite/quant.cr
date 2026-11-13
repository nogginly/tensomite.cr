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
    # 2. For each output, sums the blocks in order. A block's integer dot
    #    product is taken in wrapping arithmetic, which LLVM vectorises; for
    #    `Q4_0` it is `Σq·y - 8·Σy`, the same integer as the reference's
    #    `Σ(q - 8)·y`. Scales are read through a table of every `F16`'s value.
    #    The floating-point expressions after it are the reference's.
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
        quantize_activations(x[row * k, k], scales, sums, qs, with_sums: dtype.q4_1?)
        blocks.times { |block| totals[block] = block_total(qs.to_unsafe + block * 32) }
        out = dst[row * n, n]
        Quant.in_parallel(n, n * row_bytes) do |first, stop|
          (first...stop).each do |col|
            weight = w.to_unsafe + col * row_bytes
            out[col] = case dtype
                       when .q8_0? then fast_q8_0(weight, scales, qs, blocks)
                       when .q4_0? then fast_q4_0(weight, scales, totals, qs, blocks)
                       when .q4_1? then fast_q4_1(weight, scales, sums, qs, blocks)
                       else             raise DTypeError.new("#{dtype} is not a block format")
                       end
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
      HALVES.size
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

    # Returns the sum of a block's 32 quantised values.
    private def block_total(qb : Int8*) : Int32
      total = 0
      32.times { |j| total &+= qb[j].to_i32 }
      total
    end

    # The value of every `F16`, by its bits.
    private HALVES = Slice(Float32).new(65536) { |bits| F16.from_bits(bits.to_u16!).to_f32 }

    private def fast_half(at : UInt8*) : Float32
      HALVES.unsafe_fetch(at[0].to_i32 | at[1].to_i32 << 8)
    end

    private def fast_q8_0(w : UInt8*, scales : Slice(Float32), qs : Slice(Int8), blocks : Int32) : Float32
      sum = 0.0_f32
      sp, qp = scales.to_unsafe, qs.to_unsafe
      blocks.times do |block|
        base = w + block * 34
        values = (base + 2).as(Int8*)
        qb = qp + block * 32
        total = 0
        32.times { |j| total &+= values[j].to_i32 &* qb[j].to_i32 }
        sum += total.to_f32 * (fast_half(base) * sp[block])
      end
      sum
    end

    # Returns the integer dot product of a block's 32 nibbles, at *packed*, low
    # halves first, with its 32 quantised activations: the nibbles are
    # unpacked into bytes, then multiplied in one loop LLVM vectorises.
    private def nibble_dot(packed : UInt8*, qb : Int8*) : Int32
      values = uninitialized StaticArray(Int8, 32)
      16.times do |j|
        values[j] = (packed[j] & 0x0f).to_i8!
        values[j + 16] = (packed[j] >> 4).to_i8!
      end
      total = 0
      32.times { |j| total &+= values[j].to_i32 &* qb[j].to_i32 }
      total
    end

    private def fast_q4_0(w : UInt8*, scales : Slice(Float32), totals : Slice(Int32), qs : Slice(Int8), blocks : Int32) : Float32
      sum = 0.0_f32
      sp, tp, qp = scales.to_unsafe, totals.to_unsafe, qs.to_unsafe
      blocks.times do |block|
        base = w + block * 18
        qb = qp + block * 32
        total = nibble_dot(base + 2, qb)
        sum += (total &- 8 &* tp[block]).to_f32 * fast_half(base) * sp[block]
      end
      sum
    end

    private def fast_q4_1(w : UInt8*, scales : Slice(Float32), sums : Slice(Float32), qs : Slice(Int8), blocks : Int32) : Float32
      sum = 0.0_f32
      sp, mp, qp = scales.to_unsafe, sums.to_unsafe, qs.to_unsafe
      blocks.times do |block|
        base = w + block * 20
        qb = qp + block * 32
        total = nibble_dot(base + 4, qb)
        sum += (fast_half(base) * sp[block]) * total.to_f32 + fast_half(base + 2) * mp[block]
      end
      sum
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
        qb = qp + block * 32
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
        qb = qp + block * 32
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
        qb = qp + block * 32
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
