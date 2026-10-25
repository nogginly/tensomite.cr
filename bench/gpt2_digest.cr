# Prints a SHA-256 digest of GPT-2's logits on each device: a prefill of up to
# 64 tokens, then 8 tokens decoded one at a time. Two builds that print the same
# digests compute the same bits.
#
#     crystal run --release bench/gpt2_digest.cr -- path/to/model.safetensors [heads]
require "digest/sha256"
require "../src/tensomite/blas"
{% if flag?(:darwin) && flag?(:aarch64) %}
  require "../src/tensomite/metal"
{% end %}

include Tensomite

path = ARGV[0]? || abort "usage: bench/gpt2_digest.cr MODEL.safetensors [HEADS]"
heads = (ARGV[1]? || "12").to_i

devices = [{"CPU", CPU.default.as(Device)}, {"BLAS", BLAS.default.as(Device)}]
{% if flag?(:darwin) && flag?(:aarch64) %}
  devices << {"Metal", Metal.default.as(Device)} if Metal.available?
{% end %}

devices.each do |label, device|
  model = GPT2.load(path, heads, device)
  prompt = Array.new(Math.min(64, model.config.context - 8)) { |i| (i * 37) % model.config.vocab }
  cache = model.new_cache
  digest = Digest::SHA256.new
  logits = model.forward(prompt, cache)
  8.times do |i|
    values = logits.to_a(Float32)
    digest.update(Slice.new(values.to_unsafe.as(UInt8*), values.size * 4))
    logits = model.forward([(i * 101) % model.config.vocab], cache)
  end
  puts "#{label}: #{digest.hexfinal}"
end
