require "../../../src/tensomite"

module Server
  # A request a client may get wrong, such as an empty prompt; the server answers
  # it with status 400.
  class RequestError < Exception
  end

  # What a completion asks for, with OpenAI's defaults where it has them.
  record Request,
    prompt : String,
    max_tokens : Int32 = 16,
    temperature : Float64 = 1.0,
    top_k : Int32 = 0,
    stop : Array(String) = [] of String,
    seed : Int32 = 0

  # A finished completion: its text, the tokens it read and wrote, and why it
  # ended, `"stop"` for the end-of-text token or a stop string and `"length"` for
  # the token limit or the end of the context.
  record Completion,
    text : String,
    prompt_tokens : Int32,
    completion_tokens : Int32,
    finish_reason : String

  # Runs completions on one GPT-2 model, one request at a time.
  #
  # ```
  # engine = Server::Engine.load("models/gpt2", heads: 12, device: Tensomite::CPU.default)
  # engine.complete(Server::Request.new("Once upon a time", max_tokens: 20)) { |text| print text }
  # ```
  class Engine
    # The model's name, as completions report it.
    getter name : String

    def initialize(@model : Tensomite::GPT2, @tokenizer : Tensomite::BPE, @name : String)
      @end_of_text = tokenizer.id?("<|endoftext|>")
      @lock = Mutex.new
    end

    # Returns an engine for the model in *dir*: `model.safetensors`, `vocab.json`
    # and `merges.txt`, as Hugging Face's GPT-2 repositories hold them, named after
    # the directory.
    def self.load(dir : String, heads : Int32, device : Tensomite::Device) : Engine
      model = Tensomite::GPT2.load(File.join(dir, "model.safetensors"), heads: heads, device: device)
      tokenizer = Tensomite::BPE.from_files(File.join(dir, "vocab.json"), File.join(dir, "merges.txt"))
      new(model, tokenizer, File.basename(File.expand_path(dir)))
    end

    # Runs *request* and returns its completion, yielding its text as it is
    # written, in pieces whose concatenation is the completion's text. A request
    # waits for any other to finish.
    #
    # Text is yielded only up to the last whole character, and held back while it
    # could still be the start of a stop string. Raises `RequestError` for an
    # empty prompt or one that fills the context.
    def complete(request : Request, & : String ->) : Completion
      @lock.synchronize { run(request) { |text| yield text } }
    end

    # :ditto:
    def complete(request : Request) : Completion
      complete(request) { }
    end

    # 1. Prefills the prompt, then decodes a token at a time, up to the token limit
    #    or the end of the context.
    # 2. Keeps the bytes written so far, of which the text is the longest whole
    #    UTF-8 prefix, and ends at the first stop string in it.
    # 3. Yields the text not yet yielded, less the tail that could begin a stop
    #    string.
    private def run(request : Request, & : String ->) : Completion
      prompt = @tokenizer.encode(request.prompt)
      raise RequestError.new("the prompt is empty") if prompt.empty?
      context = @model.config.context
      if prompt.size >= context
        raise RequestError.new("the prompt's #{prompt.size} tokens leave no room in the context of #{context}")
      end
      limit = Math.min(request.max_tokens, context - prompt.size)
      sampler = Tensomite::Sampler.new(temperature: request.temperature, top_k: request.top_k, seed: request.seed)
      held = request.stop.max_of?(&.size).try { |size| size - 1 } || 0
      cache = @model.new_cache
      logits = @model.forward(prompt, cache)
      bytes = IO::Memory.new
      text, sent, count, reason, cut_at_stop = "", 0, 0, "length", false
      while count < limit
        token = sampler.pick(logits.slice(0, logits.shape[0] - 1, logits.shape[0]).to_a(Float32))
        if token == @end_of_text
          reason = "stop"
          break
        end
        count += 1
        bytes.write(@tokenizer.decode_bytes({token}))
        text = Server.whole_characters(bytes.to_slice)
        if cut = Server.first_stop(text, request.stop)
          text, reason, cut_at_stop = text[0, cut], "stop", true
          break
        end
        ready = Math.max(sent, text.size - held)
        yield text[sent...ready] if ready > sent
        sent = ready
        logits = @model.forward([token], cache) if count < limit
      end
      # A character cut by the end of the completion becomes U+FFFD.
      text = String.new(bytes.to_slice).scrub unless cut_at_stop
      yield text[sent..] if text.size > sent
      Completion.new(text, prompt.size, count, reason)
    end
  end

  # Returns the longest prefix of *bytes* that ends on a whole UTF-8 character, as
  # text.
  def self.whole_characters(bytes : Bytes) : String
    (0..Math.min(3, bytes.size)).each do |cut|
      text = String.new(bytes[0, bytes.size - cut])
      return text if text.valid_encoding?
    end
    String.new(bytes).scrub
  end

  # Returns where the first of *stops* begins in *text*, or `nil` if none does.
  def self.first_stop(text : String, stops : Array(String)) : Int32?
    stops.compact_map { |stop| text.index(stop) unless stop.empty? }.min?
  end
end
