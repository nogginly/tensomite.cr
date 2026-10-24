# Compares the naive CPU matmul with BLAS and, on an Apple Silicon Mac with a GPU,
# with Metal: first on square F32 matrices, then BLAS and Metal on the products a
# 128-token GPT-2 small prefill runs.
#
#     crystal run --release bench/matmul.cr
require "../src/tensomite/blas"
{% if flag?(:darwin) && flag?(:aarch64) %}
  require "../src/tensomite/metal"
{% end %}

include Tensomite

# Returns the median of *runs* timings of the block, in milliseconds.
def time_ms(runs : Int32, &) : Float64
  times = Array(Float64).new(runs) do
    start = Time.instant
    yield
    (Time.instant - start).total_milliseconds
  end
  times.sort[runs // 2]
end

def matrix(device : Device, n : Int32, seed : Int32) : Tensor
  matrix(device, [n, n], seed)
end

def matrix(device : Device, shape : Array(Int32), seed : Int32) : Tensor
  rng = Random.new(seed)
  Tensor.from(Array(Float32).new(shape.product(1)) { rng.rand(-1f32...1f32) }, shape, device)
end

gpu = nil.as(Device?)
{% if flag?(:darwin) && flag?(:aarch64) %}
  gpu = Metal.default if Metal.available?
{% end %}

puts "size |naive CPU ms|speedup|BLAS ms |GFLOP/s BLAS|Metal ms|GFLOP/s Metal"
puts "-----|------------|-------|--------|------------|--------|-------------"
[64, 128, 256, 512, 1024].each do |size|
  runs = size <= 128 ? 9 : (size >= 1024 ? 3 : 5)
  a, b = matrix(CPU.default, size, 1), matrix(CPU.default, size, 2)
  naive = size <= 512 ? time_ms(runs) { a.matmul(b).realize } : nil
  a, b = matrix(BLAS.default, size, 1), matrix(BLAS.default, size, 2)
  blas = time_ms(runs) { a.matmul(b).realize }
  gflops = 2.0 * size ** 3 / (blas / 1000) / 1e9
  metal = gpu.try do |device|
    a, b = matrix(device, size, 1), matrix(device, size, 2)
    time_ms(runs) { a.matmul(b).realize }
  end
  metal_cols = metal ? "%8.3f|%13.1f" % {metal, 2.0 * size ** 3 / (metal / 1000) / 1e9} : "     n/a|          n/a"
  naive_cols = naive ? "%12.2f|%6.0fx" % {naive, naive / blas} : "         n/a|    n/a"
  puts "%-5d|%s|%8.3f|%12.1f|%s" % {size, naive_cols, blas, gflops, metal_cols}
end

# The products of a 128-token prefill, as {name, a's shape, b's shape, whether
# b is read transposed}. The model's head reads the token embedding transposed.
GPT2_PREFILL = [
  {"attention in", [128, 768], [768, 2304], false},
  {"attention out", [128, 768], [768, 768], false},
  {"mlp in", [128, 768], [768, 3072], false},
  {"mlp out", [128, 3072], [3072, 768], false},
  {"scores", [12, 128, 64], [12, 64, 128], false},
  {"weighted values", [12, 128, 128], [12, 128, 64], false},
  {"head", [128, 768], [768, 50257], false},
  {"head, b transposed", [128, 768], [50257, 768], true},
]

# Returns the product's b as the model reads it.
def operand(device : Device, shape : Array(Int32), transposed : Bool) : Tensor
  b = matrix(device, shape, 2)
  transposed ? b.transpose : b
end

puts
{% if flag?(:darwin) && flag?(:aarch64) %}
  # Metal columns are GPU time per product, from `Metal::Profile`, without the
  # round trip each lone product would otherwise pay.
  configs = [{"split", Metal::MatMulMode::Split, Metal::MatMulTile::T64x64}] +
            Metal::MatMulTile.values.map { |tile| {tile.to_s.lchop('T'), Metal::MatMulMode::Tiled, tile} }
  puts "GPT-2 prefill product|BLAS ms|GFLOP/s BLAS|" + configs.map { |(name, _, _)| "GFLOP/s #{name}" }.join('|')
  puts "---------------------|-------|------------|" + configs.map { |(name, _, _)| "-" * (8 + name.size) }.join('|')
{% else %}
  puts "GPT-2 prefill product|BLAS ms|GFLOP/s BLAS"
  puts "---------------------|-------|------------"
{% end %}
GPT2_PREFILL.each do |(name, a_shape, b_shape, transposed)|
  flops = 2.0 * a_shape.product(1) * (transposed ? b_shape[-2] : b_shape.last)
  runs = flops > 1e9 ? 3 : 7
  a, b = matrix(BLAS.default, a_shape, 1), operand(BLAS.default, b_shape, transposed)
  blas = time_ms(runs) { a.matmul(b).realize }
  row = "%-21s|%7.3f|%12.1f" % {name, blas, flops / (blas / 1000) / 1e9}
  {% if flag?(:darwin) && flag?(:aarch64) %}
    if metal = gpu.as?(Metal)
      a, b = matrix(metal, a_shape, 1), operand(metal, b_shape, transposed)
      configs.each do |(_, mode, tile)|
        metal.matmul_mode, metal.matmul_tile = mode, tile
        a.matmul(b).realize
        profile = Metal::Profile.new(per_kernel: true)
        metal.profile = profile
        runs.times { a.matmul(b).realize }
        metal.profile = nil
        seconds = profile.kernels["MatMul"][1] / runs
        row += "|%.1f" % (flops / seconds / 1e9)
      end
      metal.matmul_mode, metal.matmul_tile = Metal::MatMulMode::Auto, Metal::MatMulTile::T64x64
    end
  {% end %}
  puts row
end
