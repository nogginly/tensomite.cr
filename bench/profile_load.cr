# What bench/profile_decode.cr and bench/profile_decode_cuda.cr share: the
# command line, the model, and the workload.
require "json"

module ProfileBench
  PROMPT   = 64
  DECODE   = 32
  PREFILLS =  4

  USAGE = "usage: PATH [HEADS] [--prefill N]: a GPT-2 model.safetensors (with its heads, 12 by default), " \
          "a Llama-family directory, or a Llama GGUF file"

  # Returns the model path, GPT-2's heads, and the prefill length, if any.
  def self.arguments : {String, Int32, Int32?}
    path = ARGV[0]? || abort USAGE
    prefill = ARGV.index("--prefill").try { |i| ARGV[i + 1]?.try(&.to_i) || abort "--prefill needs a token count" }
    heads = ARGV[1]?.try { |arg| arg.starts_with?("--") ? nil : arg.to_i } || 12
    {path, heads, prefill}
  end

  # Returns true when *path* holds a Llama-family model: a GGUF file, or a
  # directory whose config.json says so.
  def self.llama?(path : String) : Bool
    return true if path.ends_with?(".gguf")
    config = File.join(path, "config.json")
    File.exists?(config) && JSON.parse(File.read(config))["model_type"]?.try(&.as_s?) == "llama"
  end

  # Runs the workload on *model* and returns its wall seconds, calling *watch*
  # with true when the measured part starts and false when it ends: DECODE
  # tokens after a fresh prompt of PROMPT tokens (fewer if the context is
  # short), or PREFILLS prefills of *prefill*
  # tokens, each into an empty cache.
  def self.run(model : M, prefill : Int32?, &watch : Bool ->) : Float64 forall M
    vocab = model.config.vocab
    if prefill
      prompt = Array.new(prefill) { |i| (i * 37) % vocab }
      watch.call(true)
      start = Time.instant
      PREFILLS.times { model.forward(prompt, model.new_cache(prefill)) }
    else
      length = Math.min(PROMPT, model.config.context - DECODE)
      prompt = Array.new(length) { |i| (i * 37) % vocab }
      cache = model.new_cache(length + DECODE)
      model.forward(prompt, cache)
      watch.call(true)
      start = Time.instant
      DECODE.times { |i| model.forward([i], cache) }
    end
    (Time.instant - start).total_seconds.tap { watch.call(false) }
  end
end
