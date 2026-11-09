# Measures a model's perplexity on a text, to see what quantisation costs.
#
#     scripts/fetch_shakespeare.sh
#     crystal run --release bench/perplexity.cr -- --model models/smollm2-135m-instruct
#     crystal run --release bench/perplexity.cr -- --model models/smollm2-135m-instruct-gguf/SmolLM2-135M-Instruct-Q4_0.gguf
#     crystal run --release bench/perplexity.cr -- --model models/gpt2 --quantize Q8_0
#
# The text is cut into `--windows` windows of `--context` tokens, from its start.
# Each window is run from position 0, and every token after the first is scored
# by the cross-entropy of its prediction, so the result is the mean over
# `windows × (context - 1)` tokens. Perplexity is `e` to that mean, the number of
# equally likely tokens the model is, on average, choosing between.
#
# `--model` is a GPT-2 directory, a Llama-family directory, or a Llama GGUF
# file. `--quantize Q8_0` (or `Q4_0`, `Q4_1`) quantises the model's matrices
# with tensomite's own quantiser before measuring, and its embeddings to the
# same, or to `--embeddings TYPE` (`F32` keeps them).
#
# Perplexity alone can mislead: on a text a model is too sure of, noise that
# softens its predictions scores better. `--reference PATH` also loads an
# unquantised model and reports the mean KL divergence of this model's
# predictions from its, `Σ p (log p - log q)` per token, which is zero only
# when they agree:
#
#     crystal run --release bench/perplexity.cr -- --model models/gpt2 --quantize Q4_0 --reference models/gpt2
require "json"
require "option_parser"
require "../src/tensomite/blas"
{% if flag?(:darwin) && flag?(:aarch64) %}
  require "../src/tensomite/metal"
{% end %}
{% if flag?(:cuda) %}
  require "../src/tensomite/cuda"
{% end %}

include Tensomite

model_path, text_path, device_name = "models/gpt2", "data/shakespeare/input.txt", "blas"
context, windows = 128, 16
quantize = nil.as(DType?)
embeddings = nil.as(DType?)
reference_path = nil.as(String?)

OptionParser.parse do |parser|
  parser.banner = "Usage: perplexity [options]"
  parser.on("--model PATH", "GPT-2 or Llama directory, or Llama GGUF file (#{model_path})") { |path| model_path = path }
  parser.on("--text FILE", "Text to measure on (#{text_path})") { |path| text_path = path }
  parser.on("--context N", "Tokens per window (#{context})") { |value| context = value.to_i }
  parser.on("--windows N", "Windows to measure (#{windows})") { |value| windows = value.to_i }
  parser.on("--quantize TYPE", "Quantise to Q8_0, Q4_0 or Q4_1 first") { |value| quantize = DType.parse(value) }
  parser.on("--reference PATH", "Also report the KL divergence from this model's predictions") { |path| reference_path = path }
  parser.on("--embeddings TYPE", "Quantise embeddings to TYPE instead, or F32 to keep them") { |value| embeddings = DType.parse(value) }
  parser.on("--device NAME", "cpu, blas, metal or cuda (#{device_name})") { |name| device_name = name }
  parser.on("-h", "--help", "Show this help") { puts parser; exit }
end

devices = {"cpu" => -> { CPU.default.as(Device) }, "blas" => -> { BLAS.default.as(Device) }}
{% if flag?(:darwin) && flag?(:aarch64) %}
  devices["metal"] = -> { Metal.default.as(Device) }
{% end %}
{% if flag?(:cuda) %}
  devices["cuda"] = -> { CUDA.default.as(Device) }
{% end %}
device = devices[device_name]?.try(&.call) || abort "unknown device #{device_name}; this build has #{devices.keys.join(", ")}"

# Quantises *model*'s matrices to *dtype* and its embeddings to *embeddings*,
# or to *dtype* when that is `nil`; `F32` embeddings stay as they are.
def quantize_model(model : Module, dtype : DType, embeddings : DType?) : Nil
  model.quantize(dtype)
  table = embeddings || dtype
  model.modules.each { |_, part| part.quantize(table) if part.is_a?(Embedding) } if table.block?
end

# Returns a function from a `[1, context]` tensor of ids to its lazy logits,
# and the tokeniser, for the model at *path*.
def load(path : String, device : Device, quantize : DType?, embeddings : DType?) : {Proc(Tensor, Tensor), BPE}
  if path.ends_with?(".gguf")
    file = GGUF.open(path)
    model = Llama.load(file, device)
    quantize_model(model, quantize, embeddings) if quantize
    return {->(ids : Tensor) { model.call(ids) }, BPE.from_gguf(file)}
  end
  config = File.join(path, "config.json")
  json = File.exists?(config) ? JSON.parse(File.read(config)) : nil
  if json.try(&.["model_type"]?).try(&.as_s) == "llama"
    model = Llama.load(path, device)
    quantize_model(model, quantize, embeddings) if quantize
    return {->(ids : Tensor) { model.call(ids) }, BPE.from_json(File.join(path, "tokenizer.json"))}
  end
  heads = json.try(&.["n_head"]?).try(&.as_i) || 12
  gpt2 = GPT2.load(File.join(path, "model.safetensors"), heads: heads, device: device)
  quantize_model(gpt2, quantize, embeddings) if quantize
  {->(ids : Tensor) { gpt2.call(ids) }, BPE.from_files(File.join(path, "vocab.json"), File.join(path, "merges.txt"))}
end

forward, bpe = load(model_path, device, quantize, embeddings)
reference = reference_path.try { |path| load(path, device, nil, nil)[0] }
tokens = bpe.encode(File.read(text_path), special: false)
abort "#{text_path} has #{tokens.size} tokens, fewer than #{windows} windows of #{context}" if tokens.size < windows * context

start = Time.instant
total = divergence = 0.0
windows.times do |window|
  ids = tokens[window * context, context]
  inputs = Tensor.from(ids[0, context - 1], [1, context - 1], device)
  targets = Tensor.from(ids[1, context - 1], [context - 1], device)
  logits = forward.call(inputs)
  total += logits.cross_entropy(targets).to_a(Float32)[0]
  if reference
    log_p = reference.call(inputs).log_softmax
    divergence += (log_p.exp * (log_p - logits.log_softmax)).sum(-1).mean.to_a(Float32)[0]
  end
  GC.collect
end
seconds = (Time.instant - start).total_seconds
mean = total / windows
name = quantize ? "#{model_path} at #{quantize}#{", embeddings #{embeddings}" if embeddings}" : model_path
kl = reference ? ", KL divergence #{(divergence / windows).round(5)} nats/token" : ""
puts "#{name}: #{mean.round(4)} nats/token, perplexity #{Math.exp(mean).round(3)}#{kl}, " \
     "#{windows} windows of #{context} tokens in #{seconds.round(1)} s"
