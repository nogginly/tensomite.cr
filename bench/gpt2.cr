# Measures GPT-2 tokens per second on the CPU oracle and on BLAS.
#
#     crystal run --release bench/gpt2.cr -- path/to/model.safetensors [heads]
#
# Speed depends only on the model's shapes, so any weights of GPT-2 small's shapes
# give the same numbers as the real ones.
require "../src/tensomite/blas"

include Tensomite

PROMPT = 64
DECODE = 32

path = ARGV[0]? || abort "usage: bench/gpt2.cr MODEL.safetensors [HEADS]"
heads = (ARGV[1]? || "12").to_i

puts "device|prefill tokens/s|decode tokens/s|ms per decoded token"
puts "------|----------------|---------------|--------------------"
[CPU.default, BLAS.default].each do |device|
  model = GPT2.load(path, heads, device)
  prompt = Array.new(PROMPT) { |i| (i * 37) % model.config.vocab }
  cache = model.new_cache

  start = Time.instant
  model.forward(prompt, cache)
  prefill = (Time.instant - start).total_seconds

  start = Time.instant
  DECODE.times { |i| model.forward([i], cache) }
  decode = (Time.instant - start).total_seconds

  puts "%s|%.1f|%.2f|%.0f" % {device.class.name.split("::").last, PROMPT / prefill, DECODE / decode, decode / DECODE * 1000}
end
