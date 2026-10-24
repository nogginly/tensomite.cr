# Measures GPT-2 tokens per second on the CPU oracle, on BLAS and, on an Apple
# Silicon Mac with a GPU, on Metal.
#
#     crystal run --release bench/gpt2.cr -- path/to/model.safetensors [heads]
#
# Speed depends only on the model's shapes, so any weights of GPT-2 small's shapes
# give the same numbers as the real ones.
require "../src/tensomite/blas"
{% if flag?(:darwin) && flag?(:aarch64) %}
  require "../src/tensomite/metal"
{% end %}

include Tensomite

PROMPT = 64
DECODE = 32

path = ARGV[0]? || abort "usage: bench/gpt2.cr MODEL.safetensors [HEADS]"
heads = (ARGV[1]? || "12").to_i

puts "device|prefill tokens/s|decode tokens/s|ms per decoded token"
puts "------|----------------|---------------|--------------------"
# Each device to run, with a label and a setup to apply first.
runs = [{"CPU", CPU.default.as(Device), -> { }}, {"BLAS", BLAS.default.as(Device), -> { }}]
{% if flag?(:darwin) && flag?(:aarch64) %}
  if Metal.available?
    Metal::MatMulMode.values.each do |mode|
      runs << {"Metal, #{mode.to_s.downcase} matmul", Metal.default.as(Device), -> { Metal.default.matmul_mode = mode }}
    end
  end
{% end %}

runs.each do |label, device, setup|
  setup.call
  model = GPT2.load(path, heads, device)
  prompt = Array.new(PROMPT) { |i| (i * 37) % model.config.vocab }
  # Prefill is timed warm, as the median of five, except on the slow CPU oracle.
  repeats = device.is_a?(CPU) ? 1 : 5
  model.forward(prompt, model.new_cache) if repeats > 1
  prefills = Array.new(repeats) do
    start = Time.instant
    model.forward(prompt, model.new_cache)
    (Time.instant - start).total_seconds
  end
  prefill = prefills.sort[repeats // 2]

  cache = model.new_cache
  model.forward(prompt, cache)

  start = Time.instant
  DECODE.times { |i| model.forward([i], cache) }
  decode = (Time.instant - start).total_seconds

  puts "%s|%.1f|%.2f|%.0f" % {label, PROMPT / prefill, DECODE / decode, decode / DECODE * 1000}
end
