# Times the products of decoding one GPT-2 small token on CUDA, on the GPU alone,
# with each split block width (`CUDA::MatMulSplit`), the default choice, and two
# passes.
#
#     crystal run --release -Dcuda bench/matmul_cuda.cr
#
# Each product cycles through 12 weight matrices, one per layer as in the model,
# so that the weights come from memory rather than the GPU's cache.
require "../src/tensomite/cuda"

include Tensomite

LAYERS = 12
ROUNDS =  8

# The products of a decoded token with contiguous weights, as {name, k, n}.
SHAPES = [
  {"qkv", 768, 2304},
  {"attention out", 768, 768},
  {"mlp in", 768, 3072},
  {"mlp out", 3072, 768},
]

def matrix(device : Device, shape : Array(Int32), seed : Int32) : Tensor
  rng = Random.new(seed)
  Tensor.from(Array(Float32).new(shape.product(1)) { rng.rand(-1f32...1f32) }, shape, device)
end

abort "no CUDA GPU" unless CUDA.available?
cuda = CUDA.default
configs = CUDA::MatMulSplit.values.map { |split| {split.to_s.downcase, CUDA::MatMulMode::Split, split.as(CUDA::MatMulSplit?)} }
configs << {"default", CUDA::MatMulMode::Split, nil.as(CUDA::MatMulSplit?)}
configs << {"two pass", CUDA::MatMulMode::TwoPass, nil.as(CUDA::MatMulSplit?)}

puts "#{cuda.name}, #{cuda.gpu.multiprocessors} multiprocessors; µs per product, and GB/s of weights read"
puts
puts "product|" + configs.map(&.[0]).join("|")
puts "-------|" + configs.map { |config| "-" * config[0].size }.join("|")
SHAPES.each do |(name, k, n)|
  x = matrix(cuda, [1, k], 1)
  weights = Array.new(LAYERS) { |i| matrix(cuda, [k, n], i + 2) }
  cells = configs.map do |(_, mode, split)|
    cuda.matmul_mode, cuda.matmul_split = mode, split
    weights.each { |weight| x.matmul(weight).realize }
    profile = CUDA::Profile.new(per_kernel: true)
    cuda.profile = profile
    ROUNDS.times { weights.each { |weight| x.matmul(weight).realize } }
    cuda.profile = nil
    seconds = profile.kernels["MatMul"][1] / (ROUNDS * LAYERS)
    "%.1f, %.0f" % {seconds * 1e6, 4.0 * k * n / seconds / 1e9}
  end
  puts "#{name} [1,#{k}]·[#{k},#{n}]|#{cells.join('|')}"
end
cuda.matmul_mode, cuda.matmul_split = CUDA::MatMulMode::Auto, nil
