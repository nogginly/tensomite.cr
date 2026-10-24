# Breaks down where GPT-2's time goes on Metal, per decoded token or per prefill.
#
#     crystal run --release bench/profile_gpt2.cr -- path/to/model.safetensors [heads] [--prefill N]
#
# Without `--prefill`: 32 decoded tokens after a 64-token prompt. With it: four
# prefills of N tokens, each into an empty cache. Each workload runs twice:
#
# 1. Batched, as normal: the wall time split into encoding, allocation, waiting
#    for the GPU, host ops, and everything else (graph building, scheduling and
#    reading the logits back).
# 2. Per kernel: each op committed and timed on the GPU alone, which slows the
#    wall time but shows the GPU time per op. A prefill's ops are listed by shape.
#
# Apple Silicon only.
require "../src/tensomite"
{% if flag?(:darwin) && flag?(:aarch64) %}
  require "../src/tensomite/metal"
{% end %}

include Tensomite

PROMPT   = 64
DECODE   = 32
PREFILLS =  4

path = ARGV[0]? || abort "usage: bench/profile_gpt2.cr MODEL.safetensors [HEADS] [--prefill N]"
prefill = ARGV.index("--prefill").try { |i| ARGV[i + 1]?.try(&.to_i) || abort "--prefill needs a token count" }
heads = ARGV[1]?.try { |arg| arg.starts_with?("--") ? nil : arg.to_i } || 12

# Runs the workload with *profile* collecting and returns its wall seconds:
# DECODE tokens after a fresh prompt, or PREFILLS prefills of *prefill* tokens.
def run(model : GPT2, metal : Metal, profile : Metal::Profile, prefill : Int32?) : Float64
  if prefill
    prompt = Array.new(prefill) { |i| (i * 37) % model.config.vocab }
    metal.profile = profile
    start = Time.instant
    PREFILLS.times { model.forward(prompt, model.new_cache) }
  else
    prompt = Array.new(PROMPT) { |i| (i * 37) % model.config.vocab }
    cache = model.new_cache
    model.forward(prompt, cache)
    metal.profile = profile
    start = Time.instant
    DECODE.times { |i| model.forward([i], cache) }
  end
  (Time.instant - start).total_seconds.tap { metal.profile = nil }
end

{% if flag?(:darwin) && flag?(:aarch64) %}
  abort "no Metal GPU" unless Metal.available?
  metal = Metal.default
  model = GPT2.load(path, heads, metal)
  runs = prefill ? PREFILLS : DECODE
  unit = prefill ? "prefill of #{prefill} tokens" : "decoded token"
  ms = ->(seconds : Float64) { "%.2f" % (seconds * 1000 / runs) }
  run(model, metal, Metal::Profile.new, prefill)

  batched = Metal::Profile.new
  wall = run(model, metal, batched, prefill)
  host = batched.host_ops.values.sum(&.[1])
  other = wall - batched.encode - batched.alloc - batched.wait - host
  puts "Per #{unit}, batched (#{metal.name}):"
  puts
  puts "part|ms"
  puts "----|--"
  puts "wall|#{ms.call(wall)}"
  puts "encoding dispatches|#{ms.call(batched.encode)}"
  puts "allocating buffers|#{ms.call(batched.alloc)}"
  puts "waiting for the GPU|#{ms.call(batched.wait)}"
  puts "host ops|#{ms.call(host)}"
  puts "everything else|#{ms.call(other)}"
  puts
  puts "dispatches: #{batched.dispatches // runs}; command buffers: #{batched.command_buffers / runs}"
  puts "prefill tokens/s: #{(prefill * runs / wall).round(1)}" if prefill
  batched.host_ops.each { |name, (count, seconds)| puts "host op #{name}: #{count // runs} each, #{ms.call(seconds)} ms" }

  timed = Metal::Profile.new(per_kernel: true, shapes: !prefill.nil?)
  run(model, metal, timed, prefill)
  gpu = timed.kernels.values.sum(&.[1])
  puts
  puts "GPU time per #{unit}, each op timed alone (total #{ms.call(gpu)} ms):"
  puts
  puts "op|count|GPU ms|share|GFLOP/s"
  puts "--|-----|------|-----|-------"
  timed.kernels.to_a.sort_by { |(_, (_, seconds))| -seconds }.first(20).each do |name, (count, seconds)|
    rate = if prefill && (dims = name.scan(/\d+/).map(&.[0].to_i)) && name.starts_with?("MatMul") && dims.size >= 4
             # [..., M, K]·[..., K, N]: 2·M·K·N per batch matrix.
             a, b = dims[0, dims.size // 2], dims[dims.size // 2..]
             flops = 2.0 * a.product(1) * b.last
             "%.1f" % (flops * count / seconds / 1e9)
           else
             ""
           end
    puts "#{name}|#{count // runs}|#{ms.call(seconds)}|#{(seconds / gpu * 100).round}%|#{rate}"
  end
{% else %}
  puts "bench/profile_gpt2.cr measures Metal, which needs macOS on Apple Silicon"
{% end %}
