# Breaks down where a decoded GPT-2 token's time goes on Metal.
#
#     crystal run --release bench/profile_gpt2.cr -- path/to/model.safetensors [heads]
#
# Two passes over the same 32 decoded tokens, after a 64-token prompt:
#
# 1. Batched, as normal: the wall time split into encoding, allocation, waiting
#    for the GPU, host ops, and everything else (graph building, scheduling and
#    reading the logits back).
# 2. Per kernel: each op committed and timed on the GPU alone, which slows the
#    wall time but shows the GPU time per op.
#
# Apple Silicon only.
require "../src/tensomite"
{% if flag?(:darwin) && flag?(:aarch64) %}
  require "../src/tensomite/metal"
{% end %}

include Tensomite

PROMPT = 64
DECODE = 32

path = ARGV[0]? || abort "usage: bench/profile_gpt2.cr MODEL.safetensors [HEADS]"
heads = (ARGV[1]? || "12").to_i

# Decodes DECODE tokens after a fresh prompt with *profile* collecting, and
# returns the wall seconds of the decode alone.
def decode(model : GPT2, metal : Metal, profile : Metal::Profile) : Float64
  prompt = Array.new(PROMPT) { |i| (i * 37) % model.config.vocab }
  cache = model.new_cache
  model.forward(prompt, cache)
  metal.profile = profile
  start = Time.instant
  DECODE.times { |i| model.forward([i], cache) }
  (Time.instant - start).total_seconds.tap { metal.profile = nil }
end

def ms(seconds : Float64) : String
  "%.2f" % (seconds * 1000 / DECODE)
end

{% if flag?(:darwin) && flag?(:aarch64) %}
  abort "no Metal GPU" unless Metal.available?
  metal = Metal.default
  model = GPT2.load(path, heads, metal)
  decode(model, metal, Metal::Profile.new)

  batched = Metal::Profile.new
  wall = decode(model, metal, batched)
  host = batched.host_ops.values.sum(&.[1])
  other = wall - batched.encode - batched.alloc - batched.wait - host
  puts "Per decoded token, batched (#{metal.name}):"
  puts
  puts "part|ms"
  puts "----|--"
  puts "wall|#{ms(wall)}"
  puts "encoding dispatches|#{ms(batched.encode)}"
  puts "allocating buffers|#{ms(batched.alloc)}"
  puts "waiting for the GPU|#{ms(batched.wait)}"
  puts "host ops|#{ms(host)}"
  puts "everything else|#{ms(other)}"
  puts
  puts "dispatches per token: #{batched.dispatches // DECODE}; command buffers per token: #{batched.command_buffers / DECODE}"
  batched.host_ops.each { |name, (count, seconds)| puts "host op #{name}: #{count // DECODE} per token, #{ms(seconds)} ms" }

  timed = Metal::Profile.new(per_kernel: true)
  decode(model, metal, timed)
  gpu = timed.kernels.values.sum(&.[1])
  puts
  puts "GPU time per decoded token, each op timed alone (total #{ms(gpu)} ms):"
  puts
  puts "op|per token|GPU ms|share"
  puts "--|---------|------|-----"
  timed.kernels.to_a.sort_by { |(_, (_, seconds))| -seconds }.each do |name, (count, seconds)|
    puts "#{name}|#{count // DECODE}|#{ms(seconds)}|#{(seconds / gpu * 100).round}%"
  end
{% else %}
  puts "bench/profile_gpt2.cr measures Metal, which needs macOS on Apple Silicon"
{% end %}
