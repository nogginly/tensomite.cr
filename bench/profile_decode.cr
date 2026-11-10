# Breaks down where a model's time goes on Metal, per decoded token or per
# prefill: GPT-2, or a Llama-family model, quantised or not.
#
#     crystal run --release bench/profile_decode.cr -- path/to/gpt2/model.safetensors [heads] [--prefill N]
#     crystal run --release bench/profile_decode.cr -- models/smollm2-135m-instruct-gguf/SmolLM2-135M-Instruct-Q4_0.gguf
#     crystal run --release bench/profile_decode.cr -- models/smollm2-1.7b-instruct --quantize Q4_0
#
# `--quantize TYPE` quantises the matrices and embeddings with tensomite's own
# quantiser after loading.
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
# Both print `Metal::Profile.report`, as `bench/profile_decode_cuda.cr` prints
# `CUDA::Profile.report`.
#
# Apple Silicon only.
require "../src/tensomite"
require "./profile_load"
{% if flag?(:darwin) && flag?(:aarch64) %}
  require "../src/tensomite/metal"
{% end %}

include Tensomite

path, heads, prefill, quantize = ProfileBench.arguments

{% if flag?(:darwin) && flag?(:aarch64) %}
  # Warms the model up, then profiles it as normal and per kernel, and prints the report.
  def profile(model : M, metal : Metal, prefill : Int32?) : Nil forall M
    runs = prefill ? ProfileBench::PREFILLS : ProfileBench::DECODE
    unit = prefill ? "prefill of #{prefill} tokens" : "decoded token"
    ProfileBench.run(model, prefill) { }
    normal = Metal::Profile.new
    wall = ProfileBench.run(model, prefill) { |measuring| metal.profile = measuring ? normal : nil }
    timed = Metal::Profile.new(per_kernel: true, shapes: true)
    ProfileBench.run(model, prefill) { |measuring| metal.profile = measuring ? timed : nil }
    Metal::Profile.report(STDOUT, metal.name, unit, runs, wall, normal, timed)
    puts "\nprefill tokens/s: #{(prefill * runs / wall).round(1)}" if prefill
  end

  abort "no Metal GPU" unless Metal.available?
  metal = Metal.default
  if ProfileBench.llama?(path)
    profile(ProfileBench.quantize(Llama.load(path, metal), quantize), metal, prefill)
  else
    profile(ProfileBench.quantize(GPT2.load(path, heads, metal), quantize), metal, prefill)
  end
{% else %}
  puts "bench/profile_decode.cr measures Metal, which needs macOS on Apple Silicon"
{% end %}
