# Times batched decoding (`Generative#forward_last` over a `BatchCache`)
# against the number of rows, on a Llama-shaped model of about 20M parameters
# with random weights: each row is prefilled with a prompt of its own length,
# then decodes one token a step. Tokens a second count every row's token; the
# last column compares a step with decoding the rows one after another.
#
#     crystal run --release bench/batched_decode.cr                 # CPU
#     crystal run --release bench/batched_decode.cr -- metal        # Metal
#     crystal run --release bench/batched_decode.cr -- blas
#     crystal run --release -Dcuda bench/batched_decode.cr -- cuda
require "../src/tensomite"
require "../src/tensomite/blas"
{% if flag?(:darwin) && flag?(:aarch64) %}
  require "../src/tensomite/metal"
{% end %}
{% if flag?(:cuda) %}
  require "../src/tensomite/cuda"
{% end %}

include Tensomite

ROWS   = [1, 8, 32, 64]
STEPS  = 8
CONFIG = Llama::Config.new(layers: 8, heads: 8, kv_heads: 8, width: 512, hidden: 1408, head_width: 64,
  context: 512, vocab: 8192)

device = case ARGV.first?
         when "blas" then BLAS.default.as(Device)
         when "metal"
           {% if flag?(:darwin) && flag?(:aarch64) %}
             Metal.default.as(Device)
           {% else %}
             abort "Metal needs macOS on Apple Silicon"
           {% end %}
         when "cuda"
           {% if flag?(:cuda) %}
             abort "no CUDA GPU" unless CUDA.available?
             CUDA.default.as(Device)
           {% else %}
             abort "CUDA needs a build with -Dcuda"
           {% end %}
         else CPU.default.as(Device)
         end
model = Llama.new(CONFIG, Builder.new(rng: Random.new(1), device: device)).merge_projections
puts "#{(model.params.sum(&.value.size) / 1e6).round(1)}M parameters on #{device.class}"
puts
puts "rows|ms a step|tokens/s|against one row at a time"
puts "----|---------|--------|--------------------------"
single = 0.0
ROWS.each do |rows|
  cache = model.new_batch_cache(rows, capacity: 256)
  prompts = Array.new(rows) { |row| Array.new(16 + 7 * (row % 9)) { |i| (row * 31 + i) % CONFIG.vocab } }
  model.forward_last(prompts, cache)
  step = Array.new(rows) { |row| [row % CONFIG.vocab] }
  model.forward_last(step, cache)
  start = Time.instant
  STEPS.times { model.forward_last(step, cache) }
  seconds = (Time.instant - start).total_seconds
  single = seconds if rows == 1
  puts "#{rows}|#{(seconds * 1000 / STEPS).round(2)}|#{(rows * STEPS / seconds).round(0).to_i}|#{(single * rows / seconds).round(1)}×"
end
