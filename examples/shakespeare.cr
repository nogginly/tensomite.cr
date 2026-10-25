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

# A transformer block: attention, then an MLP, each after a layer norm and added
# back to the residual stream.
class Block < Tensomite::Module
  def initialize(width : Int32, heads : Int32, residual : Init, rng : Random, device : Device)
    @ln1 = LayerNorm.new(width, device: device)
    @attention = CausalSelfAttention.new(width, heads, bias: false, proj_init: residual, rng: rng, device: device)
    @ln2 = LayerNorm.new(width, device: device)
    @mlp = Sequential.new(
      Linear.new(width, 4 * width, bias: false, init: Init.normal(0.02), rng: rng, device: device),
      Activation.gelu,
      Linear.new(4 * width, width, bias: false, init: residual, rng: rng, device: device))
  end

  def forward(x : Tensor, mask : Tensor) : Tensor
    x = x + @attention.call(@ln1.call(x), mask)
    x + @mlp.call(@ln2.call(x))
  end
end

# A GPT with learned positions and tied input and output embeddings, without
# dropout or biases in the attention and MLP projections.
class CharGPT < Tensomite::Module
  getter layers : Int32
  getter heads : Int32
  getter width : Int32
  getter context : Int32
  getter vocab : Int32

  @blocks : Array(Block)

  def initialize(@layers, @heads, @width, @context, @vocab, device : Device, seed : Int32)
    rng = Random.new(seed)
    @tokens = Embedding.new(vocab, width, rng: rng, device: device)
    @positions = Parameter.new(Init.normal(0.02).tensor([context, width], rng, device))
    residual = Init.normal(0.02 / Math.sqrt(2 * layers))
    @blocks = Array.new(layers) { Block.new(width, heads, residual, rng, device) }
    @ln = LayerNorm.new(width, device: device)
  end

  # Returns lazy logits `[batch * length, vocab]` for *ids*, `[batch, length]`.
  def forward(ids : Tensor) : Tensor
    batch, length = ids.shape
    x = Tensomite.scope("embedding") do
      (@tokens.call(ids) + @positions.value.slice(0, 0, length)).reshape([batch * length, width])
    end
    mask = Tensor.causal_mask(length, length, ids.device).named("causal mask")
    @blocks.each { |block| x = block.call(x, mask) }
    Tensomite.scope("head") { @ln.call(x).matmul(@tokens.weight.value.transpose) }
  end
end

steps, batch, layers, heads, width, context = 2000, 16, 4, 4, 128, 64
lr, min_lr, warmup, eval_every, eval_batches, seed = 1e-3, 1e-4, 100, 250, 20, 1337
device_name, data_path, sample_length = "blas", "data/shakespeare/input.txt", 300
diagram_depth = nil.as(Int32?)
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
  parser.on("--diagram DEPTH", "Print a Mermaid summary of the loss's graph at DEPTH, or every op at 0, and exit") { |value| diagram_depth = value.to_i }
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
  {Tensor.from(inputs, [batch, context], device).named("ids"), Tensor.from(targets, [batch * context], device).named("targets")}
end

model = CharGPT.new(layers, heads, width, context, chars.size, device, seed)
count = model.params.sum(&.value.size)
puts "#{chars.size} characters, #{train.size} training and #{val.size} validation"
puts "#{layers} layers, #{heads} heads, width #{width}, context #{context}, batch #{batch}: #{count} params on #{device_name}"

if depth = diagram_depth
  inputs, targets = sample_batch.call(train)
  logits = model.call(inputs)
  loss = Tensomite.scope("loss") { logits.cross_entropy(targets) }
  puts depth > 0 ? loss.to_mermaid(model: model, depth: depth) : loss.to_mermaid(max_nodes: 400)
  exit
end

# Weight decay applies to the weight matrices and embeddings, not to the norms.
optimizer = AdamW.new(model.params, lr: lr, beta2: 0.99, weight_decay: 0.1)
# Returns the validation loss averaged over `eval_batches` batches.
evaluate = -> do
  (1..eval_batches).sum { inputs, targets = sample_batch.call(val); model.call(inputs).cross_entropy(targets).to_a(Float32)[0] } / eval_batches
end

started = Time.instant
steps.times do |step|
  progress = step < warmup ? (step + 1) / warmup : (step - warmup) / Math.max(1, steps - warmup)
  rate = step < warmup ? lr * progress : min_lr + 0.5 * (lr - min_lr) * (1 + Math.cos(Math::PI * progress))
  optimizer.lr = rate
  inputs, targets = sample_batch.call(train)
  loss_value = optimizer.step(model.call(inputs).cross_entropy(targets))
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
    logits = model.call(Tensor.from(ids, [1, ids.size], device))
    picked = sampler.pick(logits.slice(0, ids.size - 1, ids.size).to_a(Float32))
    window << picked
    print chars[picked]
  end
  print "\n"
end
