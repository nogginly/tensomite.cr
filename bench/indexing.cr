# Times ops whose cost depends on how kernels compute element offsets: dense,
# transposed and broadcast reads, reductions along and across rows, the
# gradient of a broadcast (an `Unview`) and a matmul, each well under 2³¹
# elements. A baseline for widening sizes to 64 bits (G-01, Decision EE):
# run it before and after, on the same machine and device.
#
#     crystal run --release bench/indexing.cr                 # CPU
#     crystal run --release bench/indexing.cr -- blas
#     crystal run --release bench/indexing.cr -- metal
#     crystal run --release -Dcuda bench/indexing.cr -- cuda
#
# Each case is realised WARMUP times, then RUNS times; the table gives the
# median milliseconds, and for the memory-bound cases the bytes read and
# written a second.
require "../src/tensomite"
require "../src/tensomite/blas"
{% if flag?(:darwin) && flag?(:aarch64) %}
  require "../src/tensomite/metal"
{% end %}
{% if flag?(:cuda) %}
  require "../src/tensomite/cuda"
{% end %}

include Tensomite

WARMUP =    3
RUNS   =   20
SIDE   = 4096

device = case ARGV.first?
         when "blas" then BLAS.default.as(Device)
         when "metal"
           {% if flag?(:darwin) && flag?(:aarch64) %}
             Metal.default.as(Device)
           {% else %}
             abort "Metal needs macOS on Apple Silicon"
           {% end %}
         when "cuda"
           {% if flag?(:cuda) %}
             abort "no CUDA GPU" unless CUDA.available?
             CUDA.default.as(Device)
           {% else %}
             abort "CUDA needs a build with -Dcuda"
           {% end %}
         else CPU.default.as(Device)
         end

# The plain CPU device's matmul is the oracle's, too slow at 2,048.
matmul_side = ARGV.first?.in?("blas", "metal", "cuda") ? 2048 : 512

# Returns a tensor of *shape* on *device*, with values in [-1, 1) from *seed*.
def random(device : Device, shape : Array(Int32), seed : Int32) : Tensor
  rng = Random.new(seed)
  Tensor.from(Array(Float32).new(shape.product) { rng.rand(-1f32...1f32) }, shape, device)
end

# Returns the median milliseconds of realising what the block returns.
def median_ms(& : -> Tensor) : Float64
  WARMUP.times { yield.realize }
  times = Array.new(RUNS) do
    start = Time.instant
    yield.realize
    (Time.instant - start).total_milliseconds
  end
  times.sort[RUNS // 2]
end

a = random(device, [SIDE, SIDE], 1)
b = random(device, [SIDE, SIDE], 2)
row = random(device, [SIDE], 3)
m = random(device, [matmul_side, matmul_side], 4)
n = random(device, [matmul_side, matmul_side], 5)
matrix = SIDE.to_i64 * SIDE * 4

cases = [
  {"add, dense", matrix * 3, -> { a + b }},
  {"add, one side transposed", matrix * 3, -> { a.transpose + b }},
  {"add, a broadcast row", matrix * 2, -> { a + row }},
  {"sum along rows", matrix, -> { a.sum(1) }},
  {"sum along columns (strided)", matrix, -> { a.sum(0) }},
  {"gradient of a broadcast (Unview)", matrix * 3, -> { Tensomite.grad((a * row).sum, [row]).first }},
  {"matmul #{matmul_side}³", 0_i64, -> { m.matmul(n) }},
] of {String, Int64, Proc(Tensor)}

puts "#{device.class.name.split("::").last}, #{SIDE}×#{SIDE} F32 (#{matrix // 1_000_000} MB a matrix), median of #{RUNS}:"
puts
puts "case|ms|GB/s or GFLOP/s"
puts "----|--|---------------"
cases.each do |(name, bytes, build)|
  ms = median_ms { build.call }
  rate = bytes > 0 ? "#{(bytes / ms / 1e6).round(1)} GB/s" : "#{(2.0 * matmul_side.to_f ** 3 / ms / 1e6).round(1)} GFLOP/s"
  puts "#{name}|#{ms.round(3)}|#{rate}"
end
