# Breaks down where a model's time goes on CUDA, per decoded token or per
# prefill: GPT-2, or a Llama-family model, quantised or not.
#
#     crystal run --release -Dcuda bench/profile_decode_cuda.cr -- path/to/gpt2/model.safetensors [heads] [--prefill N]
#     crystal run --release -Dcuda bench/profile_decode_cuda.cr -- models/smollm2-135m-instruct-gguf/SmolLM2-135M-Instruct-Q4_0.gguf
#
# Without `--prefill`: 32 decoded tokens after a 64-token prompt. With it: four
# prefills of N tokens, each into an empty cache. Each workload runs twice:
#
# 1. As normal: the wall time split into launching kernels, allocating, copying
#    to and from the host, waiting for the GPU, host ops, and everything else
#    (graph building and scheduling).
# 2. Per kernel: each op timed on the GPU alone between two events, which slows
#    the wall time but shows the GPU time per op. A prefill's ops are listed by
#    shape, first by op whatever the shape, then by op and shape.
#
# The GPU time of the second run, against the wall time of the first, shows how
# much of a token the GPU spends working, and how much waiting for the host.
require "../src/tensomite/cuda"
require "./profile_load"

include Tensomite

path, heads, prefill = ProfileBench.arguments

# Warms the model up, then profiles it as normal and per kernel, and prints the report.
def profile(model : M, cuda : CUDA, prefill : Int32?) : Nil forall M
  runs = prefill ? ProfileBench::PREFILLS : ProfileBench::DECODE
  unit = prefill ? "prefill of #{prefill} tokens" : "decoded token"
  ProfileBench.run(model, prefill) { }
  normal = CUDA::Profile.new
  wall = ProfileBench.run(model, prefill) { |measuring| cuda.profile = measuring ? normal : nil }
  timed = CUDA::Profile.new(per_kernel: true, shapes: true)
  ProfileBench.run(model, prefill) { |measuring| cuda.profile = measuring ? timed : nil }
  CUDA::Profile.report(STDOUT, cuda.name, unit, runs, wall, normal, timed)
  puts "\nprefill tokens/s: #{(prefill * runs / wall).round(1)}" if prefill
end

abort "no CUDA GPU" unless CUDA.available?
cuda = CUDA.default
if ProfileBench.llama?(path)
  profile(Llama.load(path, cuda), cuda, prefill)
else
  profile(GPT2.load(path, heads, cuda), cuda, prefill)
end
