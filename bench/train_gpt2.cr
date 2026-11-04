# Fine-tunes a pretrained GPT-2 on a text file for a few steps and reports what a
# step costs: milliseconds, tokens a second, and the memory its weights, gradients
# and AdamW's state take. Validation loss before and after, and a sample of each,
# show whether the model moved towards the text.
#
#     scripts/fetch_gpt2.sh && scripts/fetch_shakespeare.sh
#     crystal run --release bench/train_gpt2.cr -- --device metal --steps 20
#     crystal run --release -Dcuda bench/train_gpt2.cr -- --device cuda --profile 5
#
# `--profile N`, with `--device metal` or `cuda`, profiles N steps after three to
# warm up and exits, as `examples/shakespeare.cr --profile` does.
require "option_parser"
require "json"
require "../src/tensomite"
require "../src/tensomite/blas"
{% if flag?(:darwin) && flag?(:aarch64) %}
  require "../src/tensomite/metal"
{% end %}
{% if flag?(:cuda) %}
  require "../src/tensomite/cuda"
{% end %}

include Tensomite

model_dir, data_path, device_name = "models/gpt2", "data/shakespeare/input.txt", "blas"
batch, context, steps, lr, seed = 8, 256, 20, 1e-4, 1337
eval_batches, sample_length = 4, 40
profile_steps = nil.as(Int32?)
devices = {"cpu" => -> { CPU.default.as(Device) }, "blas" => -> { BLAS.default.as(Device) }}
{% if flag?(:darwin) && flag?(:aarch64) %}
  devices["metal"] = -> { Metal.default.as(Device) }
{% end %}
{% if flag?(:cuda) %}
  devices["cuda"] = -> { CUDA.default.as(Device) }
{% end %}
OptionParser.parse do |parser|
  parser.banner = "Usage: train_gpt2 [options]"
  parser.on("--model DIR", "Pretrained GPT-2, as scripts/fetch_gpt2.sh downloads it (#{model_dir})") { |dir| model_dir = dir }
  parser.on("--data PATH", "Text to fine-tune on (#{data_path})") { |path| data_path = path }
  parser.on("--device NAME", "#{devices.keys.join(", ")} (#{device_name})") { |name| device_name = name }
  parser.on("--batch N", "Sequences per step (#{batch})") { |value| batch = value.to_i }
  parser.on("--context N", "Tokens per sequence (#{context})") { |value| context = value.to_i }
  parser.on("--steps N", "Training steps (#{steps})") { |value| steps = value.to_i }
  parser.on("--lr X", "Learning rate (#{lr})") { |value| lr = value.to_f }
  parser.on("--seed N", "Random seed (#{seed})") { |value| seed = value.to_i }
  parser.on("--profile N", "With --device metal or cuda: profile N steps after three warm-up steps, and exit") { |value| profile_steps = value.to_i }
  parser.on("-h", "--help", "Show this help") { puts parser; exit }
end
device = devices[device_name]?.try(&.call) || abort "unknown device #{device_name}; this build has #{devices.keys.join(", ")}"

heads = JSON.parse(File.read(File.join(model_dir, "config.json")))["n_head"].as_i
model = GPT2.load(File.join(model_dir, "model.safetensors"), heads: heads, device: device)
bpe = BPE.from_files(File.join(model_dir, "vocab.json"), File.join(model_dir, "merges.txt"))
abort "--context #{context} exceeds the model's #{model.config.context}" if context > model.config.context

start = Time.instant
tokens = bpe.encode(File.read(data_path))
split = tokens.size * 9 // 10
train, val = tokens[0, split], tokens[split..]
params = model.params.sum(0_i64, &.value.size)
puts "#{File.basename(model_dir)}: #{model.config.layers} layers, width #{model.config.width}, #{(params / 1e6).round(1)}M params on #{device_name}"
puts "#{data_path}: #{tokens.size} tokens, tokenised in #{(Time.instant - start).total_seconds.round(1)} s; #{train.size} to train on"
# Weights, gradients, and AdamW's m and v, each a float per parameter.
puts "weights, gradients and AdamW state: #{(params * 4 * 4 / 1e9).round(2)} GB in F32"

rng = Random.new(seed)
# Returns inputs `[batch, context]` and next-token targets for random windows of *data*.
sample_batch = ->(data : Array(Int32)) do
  starts = Array.new(batch) { rng.rand(data.size - context - 1) }
  inputs = starts.flat_map { |offset| data[offset, context] }
  targets = starts.flat_map { |offset| data[offset + 1, context] }
  {Tensor.from(inputs, [batch, context], device), Tensor.from(targets, [batch * context], device)}
end
evaluate = -> do
  (1..eval_batches).sum { inputs, targets = sample_batch.call(val); model.call(inputs).cross_entropy(targets).to_a(Float32)[0] } / eval_batches
end
sample = -> do
  prompt = bpe.encode("ROMEO:")
  bpe.decode(model.generate(prompt, sample_length, Sampler.new(temperature: 0.8, top_k: 40, seed: seed)))
end

optimizer = AdamW.new(model.params, lr: lr, weight_decay: 0.01)
train_step = -> do
  inputs, targets = sample_batch.call(train)
  loss = optimizer.step(model.call(inputs).cross_entropy(targets))
  GC.collect
  loss
end

if count = profile_steps
  # Times *count* steps, once collecting and once timing each op alone, after
  # three to warm up, and prints the report.
  {% if flag?(:cuda) %}
    if cuda = device.as?(CUDA)
      3.times { train_step.call }
      normal, timed = CUDA::Profile.new, CUDA::Profile.new(per_kernel: true, shapes: true)
      cuda.profile = normal
      started = Time.instant
      count.times { train_step.call }
      wall = (Time.instant - started).total_seconds
      cuda.profile = timed
      count.times { train_step.call }
      cuda.profile = nil
      CUDA::Profile.report(STDOUT, cuda.name, "training step", count, wall, normal, timed, top: 25)
      exit
    end
  {% end %}
  {% if flag?(:darwin) && flag?(:aarch64) %}
    if metal = device.as?(Metal)
      3.times { train_step.call }
      normal, timed = Metal::Profile.new, Metal::Profile.new(per_kernel: true, shapes: true)
      metal.profile = normal
      started = Time.instant
      count.times { train_step.call }
      wall = (Time.instant - started).total_seconds
      metal.profile = timed
      count.times { train_step.call }
      metal.profile = nil
      Metal::Profile.report(STDOUT, metal.name, "training step", count, wall, normal, timed, top: 25)
      exit
    end
  {% end %}
  abort "--profile needs --device metal or cuda"
end

puts "before: validation loss #{evaluate.call.round(4)}"
puts "before: #{sample.call.inspect}"
times = [] of Float64
steps.times do |step|
  started = Time.instant
  loss = train_step.call
  times << (Time.instant - started).total_milliseconds
  puts "step #{step}: loss #{loss.round(4)}, #{times.last.round(1)} ms" if step % 5 == 0 || step == steps - 1
end
# The first step compiles kernels and fills the buffer pool, so the median of the
# rest is the steady cost.
steady = times.size > 1 ? times[1..].sort[(times.size - 1) // 2] : times.first
puts "median step: #{steady.round(1)} ms, #{(batch * context / (steady / 1000)).round(0)} tokens/s"
puts "after: validation loss #{evaluate.call.round(4)}"
puts "after: #{sample.call.inspect}"
