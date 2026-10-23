# Compares Metal's matmul modes (serial, deterministic split-K, atomic split-K)
# on GPT-2 small's matrix-vector shapes, and on one prefill shape for contrast.
#
#     crystal run --release bench/matvec.cr
#
# Each timing realises 20 independent products at once, so the fixed cost of
# committing a command buffer is spread over them. Apple Silicon only.
require "../src/tensomite"
{% if flag?(:darwin) && flag?(:aarch64) %}
  require "../src/tensomite/metal"
{% end %}

include Tensomite

PRODUCTS = 20
RUNS     =  9

# Rows, inner dimension and columns, named for where GPT-2 small uses them.
SHAPES = {
  "attention q·k·v, 768 → 2304" => {1, 768, 2304},
  "attention out, 768 → 768"    => {1, 768, 768},
  "MLP up, 768 → 3072"          => {1, 768, 3072},
  "MLP down, 3072 → 768"        => {1, 3072, 768},
  "logits, 768 → 50257"         => {1, 768, 50257},
  "prefill MLP up, 64 rows"     => {64, 768, 3072},
}

def random(device : Device, shape : Array(Int32), seed : Int32) : Tensor
  rng = Random.new(seed)
  Tensor.from(Array(Float32).new(shape.product(1)) { rng.rand(-1f32...1f32) }, shape, device)
end

# Returns the median time in milliseconds of realising PRODUCTS products of
# *x* by *w*, joined so that one realise runs them all.
def time_products(x : Array(Tensor), w : Tensor) : Float64
  times = Array(Float64).new(RUNS) do
    start = Time.instant
    Tensor.cat(x.map(&.matmul(w)), 0).realize
    (Time.instant - start).total_milliseconds
  end
  times.sort[RUNS // 2]
end

{% if flag?(:darwin) && flag?(:aarch64) %}
  abort "no Metal GPU" unless Metal.available?
  metal = Metal.default
  modes = Metal::MatMulMode.values
  puts "shape|" + modes.map { |mode| "#{mode} ms" }.join("|") + "|atomic saving"
  puts "-----|" + modes.map { "---" }.join("|") + "|---"
  SHAPES.each do |name, (m, k, n)|
    x = Array.new(PRODUCTS) { |i| random(metal, [m, k], i) }
    w = random(metal, [k, n], 99)
    times = modes.map do |mode|
      metal.matmul_mode = mode
      time_products(x, w)
    end
    metal.matmul_mode = Metal::MatMulMode::Split
    split, atomic = times[1], times[2]
    puts "#{name}|" + times.map { |ms| "%.3f" % ms }.join("|") + "|%.0f%%" % ((split - atomic) / split * 100)
  end
{% else %}
  puts "bench/matvec.cr measures Metal, which needs macOS on Apple Silicon"
{% end %}
