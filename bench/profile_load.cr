# What bench/profile_decode.cr and bench/profile_decode_cuda.cr share: the
# command line, the model, and the workload.
require "json"

module ProfileBench
  PROMPT   = 64
  DECODE   = 32
  PREFILLS =  4

  USAGE = "usage: PATH [HEADS] [--prefill N] [--quantize TYPE]: a GPT-2 model.safetensors (with its heads, " \
          "12 by default), a Llama-family directory, or a Llama GGUF file"

  # Returns the model path, GPT-2's heads, the prefill length, if any, and the
  # block format to quantise to, if any.
  def self.arguments : {String, Int32, Int32?, Tensomite::DType?}
    path = ARGV[0]? || abort USAGE
    prefill = option("--prefill").try(&.to_i)
    quantize = option("--quantize").try { |name| Tensomite::DType.parse(name) }
    heads = ARGV[1]?.try { |arg| arg.starts_with?("--") ? nil : arg.to_i } || 12
    {path, heads, prefill, quantize}
  end

  # Returns the value given for option *name*, as `--name VALUE`, if any.
  def self.option(name : String) : String?
    ARGV.index(name).try { |i| ARGV[i + 1]? || abort "#{name} needs a value" }
  end

  # Returns *model* with its matrices and embeddings quantised to *dtype*, or
  # as it is when *dtype* is `nil`.
  def self.quantize(model : M, dtype : Tensomite::DType?) : M forall M
    dtype ? model.quantize(dtype, embeddings: true) : model
  end

  # Returns true when *path* holds a Llama-family model: a GGUF file, or a
  # directory whose config.json says so.
  def self.llama?(path : String) : Bool
    return true if path.ends_with?(".gguf")
    config = File.join(path, "config.json")
    File.exists?(config) && JSON.parse(File.read(config))["model_type"]?.try(&.as_s?) == "llama"
  end

  # Runs the workload on *model* and returns its wall seconds, calling *watch*
  # with true when the measured part starts and false when it ends: *decode*
  # tokens after a fresh prompt of *prompt_length* tokens (fewer if the context
  # is short), or PREFILLS prefills of *prefill* tokens, each into an empty
  # cache.
  def self.run(model : M, prefill : Int32?, prompt_length : Int32 = PROMPT, decode : Int32 = DECODE, &watch : Bool ->) : Float64 forall M
    vocab = model.config.vocab
    if prefill
      prompt = Array.new(prefill) { |i| (i * 37) % vocab }
      watch.call(true)
      start = Time.instant
      PREFILLS.times { model.forward(prompt, model.new_cache(prefill)) }
    else
      length = Math.min(prompt_length, model.config.context - decode)
      prompt = Array.new(length) { |i| (i * 37) % vocab }
      cache = model.new_cache(length + decode)
      model.forward(prompt, cache)
      watch.call(true)
      start = Time.instant
      decode.times { |i| model.forward([i], cache) }
    end
    (Time.instant - start).total_seconds.tap { watch.call(false) }
  end
end
