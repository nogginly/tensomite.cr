# Compares the naive CPU matmul with BLAS on square F32 matrices.
#
#     crystal run --release bench/matmul.cr
require "../src/tensomite/blas"

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
  rng = Random.new(seed)
  Tensor.from(Array(Float32).new(n * n) { rng.rand(-1f32...1f32) }, [n, n], device)
end

puts "size |naive CPU ms|BLAS ms |speedup|GFLOP/s BLAS"
puts "-----|------------|--------|-------|------------"
[64, 128, 256, 512].each do |size|
  runs = size <= 128 ? 9 : 5
  a, b = matrix(CPU.default, size, 1), matrix(CPU.default, size, 2)
  naive = time_ms(runs) { a.matmul(b).realize }
  a, b = matrix(BLAS.default, size, 1), matrix(BLAS.default, size, 2)
  blas = time_ms(runs) { a.matmul(b).realize }
  gflops = 2.0 * size ** 3 / (blas / 1000) / 1e9
  puts "%-5d|%12.2f|%8.3f|%6.0fx|%12.1f" % {size, naive, blas, naive / blas, gflops}
end
