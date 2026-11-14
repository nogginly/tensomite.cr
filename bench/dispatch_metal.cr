# Separates the fixed cost of a Metal dispatch from the cost of a quantised
# product: a chain of small additions gives the GPU's time per dispatch, and
# Q4_0 matrix-vector products of growing size give the time beyond it.
#
#     crystal run --release bench/dispatch_metal.cr
#
# Each timing realises REPEATS rounds of its tensors at once, in the normal
# batched mode, after WARMUPS untimed runs, so that the GPU stays busy and its
# clock up, and a realise's own wait is shared among many dispatches. Apple
# Silicon only.
require "../src/tensomite"
{% if flag?(:darwin) && flag?(:aarch64) %}
  require "../src/tensomite/metal"
{% end %}

include Tensomite

RUNS     =   9
WARMUPS  =   3
REPEATS  =  10
CHAIN    = 960
PRODUCTS =  24

def random_values(count : Int32, seed : Int32) : Array(Float32)
  rng = Random.new(seed)
  Array(Float32).new(count) { rng.rand(-1f32...1f32) }
end

def random(device : Device, shape : Array(Int32), seed : Int32) : Tensor
  Tensor.from(random_values(shape.product(1), seed), shape, device)
end

# Returns the median time in microseconds of realising the tensors that the
# block builds. The block runs WARMUPS times first, and again before each
# timed run, so that only `Tensor.realize` is timed.
def median_us(& : -> Array(Tensor)) : Float64
  WARMUPS.times { Tensor.realize(yield) }
  times = Array(Float64).new(RUNS) do
    outputs = yield
    start = Time.instant
    Tensor.realize(outputs)
    (Time.instant - start).total_microseconds
  end
  times.sort[RUNS // 2]
end

# Prints a row for REPEATS rounds of PRODUCTS products of a [1, k] row by a
# Q4_0 weight of [n, k], each weight with its own buffer, and returns the weight's bytes and the
# microseconds each product took.
def product_row(device : Device, n : Int32, k : Int32) : {Float64, Float64}
  values = random_values(n * k, n + k)
  weights = Array.new(PRODUCTS) { Tensor.from(values, [n, k], device).quantize(DType::Q4_0).realize }
  x = random(device, [1, k], 7).realize
  bytes = DType::Q4_0.bytes_for(n * k).to_f64
  each = median_us { Array.new(REPEATS) { weights.map { |weight| x.linear(weight) } }.flatten } / (PRODUCTS * REPEATS)
  puts "#{n}|#{k}|%.2f|%.2f|%.0f" % {bytes / 1e6, each, bytes / (each * 1e-6) / 1e9}
  {bytes, each}
end

{% if flag?(:darwin) && flag?(:aarch64) %}
  abort "no Metal GPU" unless Metal.available?
  metal = Metal.default

  puts "Dispatch floor: a chain of #{CHAIN} dependent additions"
  puts "op|µs each"
  puts "---|---"
  x = random(metal, [1, 2048], 1).realize
  b = random(metal, [1, 2048], 2).realize
  chain = median_us do
    y = x
    CHAIN.times { y = y + b }
    [y]
  end
  puts "Add [1,2048]|%.2f" % (chain / CHAIN)
  puts

  puts "Products by size, k = 2048"
  puts "n|k|weight MB|µs each|GB/s"
  puts "---|---|---|---|---"
  rows = [64, 512, 2048, 6144, 16384].map { |n| product_row(metal, n, 2048) }
  # Least squares of µs each against bytes: the intercept is the fixed cost
  # and the slope's inverse is the rate.
  count = rows.size
  mean_bytes = rows.sum(&.[0]) / count
  mean_us = rows.sum(&.[1]) / count
  slope = rows.sum { |bytes, us| (bytes - mean_bytes) * (us - mean_us) } /
          rows.sum { |bytes, _| (bytes - mean_bytes) ** 2 }
  puts "fit: %.1f µs + bytes at %.0f GB/s" % {mean_us - slope * mean_bytes, 1e-3 / slope}
  puts

  puts "Products by row length, n = 64"
  puts "n|k|weight MB|µs each|GB/s"
  puts "---|---|---|---|---"
  [512, 2048, 8192].each { |k| product_row(metal, 64, k) }
{% else %}
  puts "bench/dispatch_metal.cr measures Metal, which needs macOS on Apple Silicon"
{% end %}
