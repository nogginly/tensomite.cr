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
#    wall time but shows the GPU time per op, first by op whatever the shape,
#    then by op and shape.
#
# Both print `Metal::Profile.report`, as `bench/profile_gpt2_cuda.cr` prints
# `CUDA::Profile.report`.
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
  run(model, metal, Metal::Profile.new, prefill)

  normal = Metal::Profile.new
  wall = run(model, metal, normal, prefill)
  timed = Metal::Profile.new(per_kernel: true, shapes: true)
  run(model, metal, timed, prefill)
  Metal::Profile.report(STDOUT, metal.name, unit, runs, wall, normal, timed)
  puts "\nprefill tokens/s: #{(prefill * runs / wall).round(1)}" if prefill
{% else %}
  puts "bench/profile_gpt2.cr measures Metal, which needs macOS on Apple Silicon"
{% end %}
