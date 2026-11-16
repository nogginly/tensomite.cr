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
#     crystal run --release bench/batched_decode.cr -- metal --profile 64
#
# With `--profile ROWS`, on Metal or CUDA, it then profiles STEPS steps of
# ROWS rows, as `bench/profile_decode.cr` profiles a token: the wall time
# split into the host's parts and waiting for the GPU, then each op timed alone.
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

# Returns the wall seconds of STEPS steps of *rows* rows, each appending one
# token to every row, after each row's prompt and one step; *watch* is called
# with true as the measured steps start and false as they end.
def steps(model : Llama, rows : Int32, & : Bool ->) : Float64
  cache = model.new_batch_cache(rows, capacity: 256)
  prompts = Array.new(rows) { |row| Array.new(16 + 7 * (row % 9)) { |i| (row * 31 + i) % CONFIG.vocab } }
  model.forward_last(prompts, cache)
  step = Array.new(rows) { |row| [row % CONFIG.vocab] }
  model.forward_last(step, cache)
  yield true
  start = Time.instant
  STEPS.times { model.forward_last(step, cache) }
  seconds = (Time.instant - start).total_seconds
  yield false
  seconds
end

model = Llama.new(CONFIG, Builder.new(rng: Random.new(1), device: device)).merge_projections
puts "#{(model.params.sum(&.value.size) / 1e6).round(1)}M parameters on #{device.class}"
puts
puts "rows|ms a step|tokens/s|against one row at a time"
puts "----|---------|--------|--------------------------"
single = 0.0
ROWS.each do |rows|
  seconds = steps(model, rows) { }
  single = seconds if rows == 1
  puts "#{rows}|#{(seconds * 1000 / STEPS).round(2)}|#{(rows * STEPS / seconds).round(0).to_i}|#{(single * rows / seconds).round(1)}×"
end

if profiled = ARGV.index("--profile").try { |i| ARGV[i + 1]?.try(&.to_i) || abort "--profile needs a number of rows" }
  unit = "step of #{profiled} rows"
  puts
  {% if flag?(:darwin) && flag?(:aarch64) %}
    if metal = device.as?(Metal)
      normal = Metal::Profile.new
      wall = steps(model, profiled) { |measuring| metal.profile = measuring ? normal : nil }
      timed = Metal::Profile.new(per_kernel: true, shapes: true)
      steps(model, profiled) { |measuring| metal.profile = measuring ? timed : nil }
      Metal::Profile.report(STDOUT, metal.name, unit, STEPS, wall, normal, timed)
    end
  {% end %}
  {% if flag?(:cuda) %}
    if cuda = device.as?(CUDA)
      normal = CUDA::Profile.new
      wall = steps(model, profiled) { |measuring| cuda.profile = measuring ? normal : nil }
      timed = CUDA::Profile.new(per_kernel: true, shapes: true)
      steps(model, profiled) { |measuring| cuda.profile = measuring ? timed : nil }
      CUDA::Profile.report(STDOUT, cuda.name, unit, STEPS, wall, normal, timed)
    end
  {% end %}
end
