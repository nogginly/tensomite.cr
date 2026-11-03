# Times Metal's tiled matmul with each `Metal::MatMulTile` on the products of a
# training step of the larger Shakespeare model, beside `Auto`, and checks each
# against `Split` bit for bit. Apple Silicon only.
#
#     crystal run --release bench/matmul_tiles.cr
require "../src/tensomite/metal"

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

def matrix(device : Device, shape : Array(Int32), seed : Int32) : Tensor
  rng = Random.new(seed)
  Tensor.from(Array(Float32).new(shape.product(1)) { rng.rand(-1f32...1f32) }, shape, device)
end

abort "needs a Metal GPU" unless Metal.available?
metal = Metal.default

# The products of a training step, as {name, a's shape, b's shape, a read
# transposed, b read transposed}: forward, then the input's and the weight's
# gradients, then attention's batched products.
PRODUCTS = [
  {"mlp in", [2048, 192], [192, 768], false, false},
  {"mlp out", [2048, 768], [768, 192], false, false},
  {"qkv", [2048, 192], [192, 576], false, false},
  {"attention out", [2048, 192], [192, 192], false, false},
  {"mlp in, dx", [2048, 768], [768, 192], false, true},
  {"mlp out, dw", [768, 2048], [2048, 192], true, false},
  {"mlp in, dw", [192, 2048], [2048, 768], true, false},
  {"qkv, dw", [192, 2048], [2048, 576], true, false},
  {"q·kᵀ", [96, 128, 32], [96, 32, 128], false, true},
  {"weights·v", [96, 128, 128], [96, 128, 32], false, false},
]

tiles = Metal::MatMulTile.values
puts "GFLOP/s of each product, and whether each tile matches `Split` bit for bit:"
puts
puts (["product", "Auto"] + tiles.map(&.to_s)).join("|")
puts (["-------", "----"] + tiles.map { "-" * 5 }).join("|")
# Returns *shape* with its last two dimensions swapped.
swapped = ->(shape : Array(Int32)) { shape[0...-2] + [shape[-1], shape[-2]] }

PRODUCTS.each do |(name, a_shape, b_shape, a_transposed, b_transposed)|
  a = a_transposed ? matrix(metal, swapped.call(a_shape), 1).transpose : matrix(metal, a_shape, 1)
  b = b_transposed ? matrix(metal, swapped.call(b_shape), 2).transpose : matrix(metal, b_shape, 2)
  flops = 2.0 * a.size * b_shape[-1]
  product = ->(mode : Metal::MatMulMode, tile : Metal::MatMulTile) do
    metal.matmul_mode, metal.matmul_tile = mode, tile
    a.matmul(b).realize
  end
  split = product.call(Metal::MatMulMode::Split, Metal::MatMulTile::T64x64).to_a(Float32)
  cells = [name]
  ([{Metal::MatMulMode::Auto, Metal::MatMulTile::T64x64}] + tiles.map { |tile| {Metal::MatMulMode::Tiled, tile} }).each do |(mode, tile)|
    same = product.call(mode, tile).to_a(Float32) == split
    ms = time_ms(9) { product.call(mode, tile) }
    cells << "%.0f%s" % {flops / (ms / 1000) / 1e9, same ? "" : " (bits differ)"}
  end
  puts cells.join("|")
end
metal.matmul_mode, metal.matmul_tile = Metal::MatMulMode::Auto, Metal::MatMulTile::T64x64
