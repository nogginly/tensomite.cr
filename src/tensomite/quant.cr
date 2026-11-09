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
    # dot products compute it.
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
    def matmul(dtype : DType, x : Slice(Float32), w : Bytes, rows : Int32, n : Int32, k : Int32, dst : Slice(Float32)) : Nil
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
