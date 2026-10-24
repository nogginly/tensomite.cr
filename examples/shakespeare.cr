# Trains a character-level GPT on Tiny Shakespeare and reports validation loss in
# nats per character, then samples from the trained model.
#
#     scripts/fetch_shakespeare.sh
#     crystal run --release examples/shakespeare.cr -- --steps 2000
#
# Run with `--help` for the model and training options.
require "option_parser"
require "../src/tensomite"
require "../src/tensomite/blas"

include Tensomite

# A GPT with learned positions and tied input and output embeddings, without
# dropout or biases in the attention and MLP projections.
class CharGPT
  getter layers : Int32
  getter heads : Int32
  getter width : Int32
  getter context : Int32
  getter vocab : Int32
  # The weights, in the order `#forward` expects; replace them after each step.
  property params : Array(Tensor)

  def initialize(@layers, @heads, @width, @context, @vocab, device : Device, seed : Int32)
    rng = Random.new(seed)
    normal = ->(shape : Array(Int32), std : Float64) do
      data = Array(Float32).new(shape.product(1)) do
        (std * Math.sqrt(-2 * Math.log(1 - rng.rand)) * Math.cos(2 * Math::PI * rng.rand)).to_f32
      end
      Tensor.from(data, shape, device)
    end
    ones = ->(n : Int32) { Tensor.full([n], 1, device: device) }
    zeros = ->(n : Int32) { Tensor.zeros([n], device: device) }
    residual_std = 0.02 / Math.sqrt(2 * layers)
    @params = [normal.call([vocab, width], 0.02), normal.call([context, width], 0.02)]
    layers.times do
      @params.concat([ones.call(width), zeros.call(width), normal.call([width, 3 * width], 0.02),
                      normal.call([width, width], residual_std),
                      ones.call(width), zeros.call(width), normal.call([width, 4 * width], 0.02),
                      normal.call([4 * width, width], residual_std)])
    end
    @params.concat([ones.call(width), zeros.call(width)])
  end

  # Returns lazy logits `[batch * length, vocab]` for *ids*, `[batch, length]`.
  def forward(ids : Tensor) : Tensor
    batch, length = ids.shape
    wte, wpe = params[0], params[1]
    x = (wte.gather(ids) + wpe.slice(0, 0, length)).reshape([batch * length, width])
    mask = Tensor.causal_mask(length, length, ids.device)
    layers.times do |layer|
      ln1_w, ln1_b, attn_w, proj_w, ln2_w, ln2_b, fc_w, out_w = params[2 + 8 * layer, 8]
      x = x + attention(x.layernorm(ln1_w, ln1_b).matmul(attn_w), batch, length, mask).matmul(proj_w)
      x = x + x.layernorm(ln2_w, ln2_b).matmul(fc_w).gelu.matmul(out_w)
    end
    x.layernorm(params[-2], params[-1]).matmul(wte.transpose)
  end

  # Returns causal self-attention `[batch * length, width]` from the joined
  # queries, keys and values, *qkv*.
  private def attention(qkv : Tensor, batch : Int32, length : Int32, mask : Tensor) : Tensor
    head_width = width // heads
    q, k, v = (0..2).map do |part|
      qkv.slice(1, part * width, (part + 1) * width)
        .reshape([batch, length, heads, head_width]).permute([0, 2, 1, 3])
    end
    scores = q.matmul(k.transpose) * (1 / Math.sqrt(head_width)) + mask
    scores.softmax.matmul(v).permute([0, 2, 1, 3]).reshape([batch * length, width])
  end
end

steps, batch, layers, heads, width, context = 2000, 16, 4, 4, 128, 64
lr, min_lr, warmup, eval_every, eval_batches, seed = 1e-3, 1e-4, 100, 250, 20, 1337
device_name, data_path, sample_length = "blas", "data/shakespeare/input.txt", 300
OptionParser.parse do |parser|
  parser.banner = "Usage: shakespeare [options]"
  parser.on("--steps N", "Training steps (#{steps})") { |value| steps = value.to_i }
  parser.on("--batch N", "Sequences per step (#{batch})") { |value| batch = value.to_i }
  parser.on("--layers N", "Transformer blocks (#{layers})") { |value| layers = value.to_i }
  parser.on("--heads N", "Attention heads (#{heads})") { |value| heads = value.to_i }
  parser.on("--width N", "Embedding width (#{width})") { |value| width = value.to_i }
  parser.on("--context N", "Characters per sequence (#{context})") { |value| context = value.to_i }
  parser.on("--lr X", "Peak learning rate (#{lr})") { |value| lr = value.to_f }
  parser.on("--min-lr X", "Final learning rate (#{min_lr})") { |value| min_lr = value.to_f }
  parser.on("--warmup N", "Warm-up steps (#{warmup})") { |value| warmup = value.to_i }
  parser.on("--eval-every N", "Steps between validation runs (#{eval_every})") { |value| eval_every = value.to_i }
  parser.on("--eval-batches N", "Batches per validation run (#{eval_batches})") { |value| eval_batches = value.to_i }
  parser.on("--sample N", "Characters to sample after training (#{sample_length})") { |value| sample_length = value.to_i }
  parser.on("--seed N", "Random seed (#{seed})") { |value| seed = value.to_i }
  parser.on("--device NAME", "cpu or blas (#{device_name})") { |name| device_name = name }
  parser.on("--data PATH", "Training text (#{data_path})") { |path| data_path = path }
  parser.on("-h", "--help", "Show this help") { puts parser; exit }
end
device = device_name == "cpu" ? CPU.default : BLAS.default

text = File.read(data_path)
chars = text.chars.uniq!.sort!
index = chars.each_with_index.to_h
encoded = text.chars.map { |char| index[char] }
split = encoded.size * 9 // 10
train, val = encoded[0, split], encoded[split..]

rng = Random.new(seed)
# Returns inputs and next-character targets for *batch* random windows of *data*.
sample_batch = ->(data : Array(Int32)) do
  starts = Array.new(batch) { rng.rand(data.size - context) }
  inputs = starts.flat_map { |start| data[start, context] }
  targets = starts.flat_map { |start| data[start + 1, context] }
  {Tensor.from(inputs, [batch, context], device), Tensor.from(targets, [batch * context], device)}
end

model = CharGPT.new(layers, heads, width, context, chars.size, device, seed)
count = model.params.sum(&.size)
puts "#{chars.size} characters, #{train.size} training and #{val.size} validation"
puts "#{layers} layers, #{heads} heads, width #{width}, context #{context}, batch #{batch}: #{count} params on #{device_name}"

decayed = model.params.map { |param| param.ndim == 2 }
matrices = AdamW.new(lr: lr, beta2: 0.99, weight_decay: 0.1)
vectors = AdamW.new(lr: lr, beta2: 0.99, weight_decay: 0.0)
# Returns the validation loss averaged over `eval_batches` batches.
evaluate = -> do
  (1..eval_batches).sum { inputs, targets = sample_batch.call(val); model.forward(inputs).cross_entropy(targets).to_a(Float32)[0] } / eval_batches
end

started = Time.instant
steps.times do |step|
  progress = step < warmup ? (step + 1) / warmup : (step - warmup) / Math.max(1, steps - warmup)
  rate = step < warmup ? lr * progress : min_lr + 0.5 * (lr - min_lr) * (1 + Math.cos(Math::PI * progress))
  matrices.lr = vectors.lr = rate
  inputs, targets = sample_batch.call(train)
  loss = model.forward(inputs).cross_entropy(targets)
  grads = Tensomite.grad(loss, model.params)
  loss_value = loss.to_a(Float32)[0]
  params = model.params
  updated_matrices = matrices.step(params.select(&.ndim.==(2)), grads.each_with_index.select { |_, i| decayed[i] }.map(&.first).to_a)
  updated_vectors = vectors.step(params.reject(&.ndim.==(2)), grads.each_with_index.reject { |_, i| decayed[i] }.map(&.first).to_a)
  model.params = decayed.map { |matrix| matrix ? updated_matrices.shift : updated_vectors.shift }
  # The step's graph is garbage now; collecting returns its buffers to the
  # device's pool before the next step allocates.
  GC.collect
  if step % 10 == 0 || step == steps - 1
    elapsed = (Time.instant - started).total_seconds
    puts "step #{step}: loss #{loss_value.round(4)}, lr #{rate.round(6)}, #{(elapsed / (step + 1) * 1000).round(1)} ms/step"
  end
  if (step + 1) % eval_every == 0 || step == steps - 1
    puts "step #{step}: validation loss #{evaluate.call.round(4)} nats/char"
  end
end

if sample_length > 0
  window = [index['\n']]
  sampler = Sampler.new(temperature: 0.8, top_k: 20, seed: seed)
  print "\n"
  sample_length.times do
    ids = window.last(context)
    logits = model.forward(Tensor.from(ids, [1, ids.size], device))
    picked = sampler.pick(logits.slice(0, ids.size - 1, ids.size).to_a(Float32))
    window << picked
    print chars[picked]
  end
  print "\n"
end
