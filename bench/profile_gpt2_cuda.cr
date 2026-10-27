# Breaks down where GPT-2's time goes on CUDA, per decoded token or per prefill.
#
#     crystal run --release -Dcuda bench/profile_gpt2_cuda.cr -- path/to/model.safetensors [heads] [--prefill N]
#
# Without `--prefill`: 32 decoded tokens after a 64-token prompt. With it: four
# prefills of N tokens, each into an empty cache. Each workload runs twice:
#
# 1. As normal: the wall time split into launching kernels, allocating, copying
#    to and from the host, waiting for the GPU, host ops, and everything else
#    (graph building and scheduling).
# 2. Per kernel: each op timed on the GPU alone between two events, which slows
#    the wall time but shows the GPU time per op. A prefill's ops are listed by
#    shape.
#
# The GPU time of the second run, against the wall time of the first, shows how
# much of a token the GPU spends working, and how much waiting for the host.
require "../src/tensomite/cuda"

include Tensomite

PROMPT   = 64
DECODE   = 32
PREFILLS =  4

path = ARGV[0]? || abort "usage: bench/profile_gpt2_cuda.cr MODEL.safetensors [HEADS] [--prefill N]"
prefill = ARGV.index("--prefill").try { |i| ARGV[i + 1]?.try(&.to_i) || abort "--prefill needs a token count" }
heads = ARGV[1]?.try { |arg| arg.starts_with?("--") ? nil : arg.to_i } || 12

# Runs the workload with *profile* collecting and returns its wall seconds:
# DECODE tokens after a fresh prompt, or PREFILLS prefills of *prefill* tokens.
def run(model : GPT2, cuda : CUDA, profile : CUDA::Profile, prefill : Int32?) : Float64
  if prefill
    prompt = Array.new(prefill) { |i| (i * 37) % model.config.vocab }
    cuda.profile = profile
    start = Time.instant
    PREFILLS.times { model.forward(prompt, model.new_cache) }
  else
    prompt = Array.new(PROMPT) { |i| (i * 37) % model.config.vocab }
    cache = model.new_cache
    model.forward(prompt, cache)
    cuda.profile = profile
    start = Time.instant
    DECODE.times { |i| model.forward([i], cache) }
  end
  (Time.instant - start).total_seconds.tap { cuda.profile = nil }
end

abort "no CUDA GPU" unless CUDA.available?
cuda = CUDA.default
model = GPT2.load(path, heads, cuda)
runs = prefill ? PREFILLS : DECODE
unit = prefill ? "prefill of #{prefill} tokens" : "decoded token"
ms = ->(seconds : Float64) { "%.3f" % (seconds * 1000 / runs) }
run(model, cuda, CUDA::Profile.new, prefill)

normal = CUDA::Profile.new
wall = run(model, cuda, normal, prefill)
host = normal.host_ops.values.sum(&.[1])
other = wall - normal.launch - normal.alloc - normal.transfer - normal.wait - host
puts "Per #{unit} (#{cuda.name}):"
puts
puts "part|ms"
puts "----|--"
puts "wall|#{ms.call(wall)}"
puts "launching kernels|#{ms.call(normal.launch)}"
puts "allocating buffers|#{ms.call(normal.alloc)}"
puts "copying to and from the host|#{ms.call(normal.transfer)}"
puts "waiting for the GPU|#{ms.call(normal.wait)}"
puts "host ops|#{ms.call(host)}"
puts "everything else|#{ms.call(other)}"
puts
puts "launches: #{normal.launches // runs}; uploads: #{normal.uploads // runs}; downloads: #{normal.downloads // runs}"
puts "prefill tokens/s: #{(prefill * runs / wall).round(1)}" if prefill
normal.host_ops.each { |name, (count, seconds)| puts "host op #{name}: #{count // runs} each, #{ms.call(seconds)} ms" }

timed = CUDA::Profile.new(per_kernel: true, shapes: true)
run(model, cuda, timed, prefill)
gpu = timed.kernels.values.sum(&.[1])
puts
puts "GPU time per #{unit}, each op timed alone (total #{ms.call(gpu)} ms, #{(gpu / wall * 100).round}% of the first run's wall time):"
puts
puts "op|count|GPU ms|share|GB/s or GFLOP/s"
puts "--|-----|------|-----|---------------"
timed.kernels.to_a.sort_by { |(_, (_, seconds))| -seconds }.first(20).each do |name, (count, seconds)|
  rate = ""
  if name.starts_with?("MatMul") && (dims = name.scan(/\d+/).map(&.[0].to_i)) && dims.size >= 4
    # [..., M, K]·[..., K, N]: 2·M·K·N flops per batch matrix, and K·N weights read.
    a, b = dims[0, dims.size // 2], dims[dims.size // 2..]
    rate = if a[-2] == 1
             "%.1f GB/s" % (4.0 * b[-2] * b[-1] * b[0...-2].product(1) * count / seconds / 1e9)
           else
             "%.1f GFLOP/s" % (2.0 * a.product(1) * b.last * count / seconds / 1e9)
           end
  end
  puts "#{name}|#{count // runs}|#{ms.call(seconds)}|#{(seconds / gpu * 100).round}%|#{rate}"
end
