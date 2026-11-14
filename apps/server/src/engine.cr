require "json"
require "../../../src/tensomite"

module Server
  # A request a client may get wrong, such as an empty prompt; the server answers
  # it with status 400.
  class RequestError < Exception
  end

  # One message of a chat: who wrote it (`system`, `user` or `assistant`) and its text.
  record Message, role : String, content : String

  # What a completion asks for, with OpenAI's defaults where it has them. With
  # *messages*, it is a chat, and *prompt* is ignored.
  record Request,
    prompt : String,
    max_tokens : Int32 = 16,
    temperature : Float64 = 1.0,
    top_k : Int32 = 0,
    stop : Array(String) = [] of String,
    seed : Int32 = 0,
    messages : Array(Message)? = nil

  # A finished completion: its text, the tokens it read and wrote, and why it
  # ended, `"stop"` for the end-of-text token or a stop string and `"length"` for
  # the token limit or the end of the context.
  record Completion,
    text : String,
    prompt_tokens : Int32,
    completion_tokens : Int32,
    finish_reason : String

  # A model as the engine runs it: its context, and sessions that each run one
  # sequence through a cache of their own.
  abstract class Decoder
    abstract def context : Int32

    # Returns a function that runs the next ids of a new sequence, with room for
    # *capacity* positions, and returns the logits of the last of them, `[1, vocab]`.
    abstract def session(capacity : Int32) : Proc(Array(Int32), Tensomite::Tensor)
  end

  # A `Decoder` for a model with `config.context`, `new_cache(capacity)` and
  # `forward_last(ids, cache)`, as `GPT2` and `Llama` have (`Tensomite::Generative`).
  class ModelDecoder(M) < Decoder
    def initialize(@model : M)
    end

    def context : Int32
      @model.config.context
    end

    def session(capacity : Int32) : Proc(Array(Int32), Tensomite::Tensor)
      cache = @model.new_cache(capacity)
      ->(ids : Array(Int32)) { @model.forward_last(ids, cache) }
    end
  end

  # Runs completions on one GPT-2 or Llama-family model, one request at a time,
  # and chats in ChatML with a model whose tokeniser has its tokens.
  #
  # ```
  # engine = Server::Engine.load("models/gpt2", heads: 12, device: Tensomite::CPU.default)
  # engine.complete(Server::Request.new("Once upon a time", max_tokens: 20)) { |text| print text }
  # ```
  class Engine
    # The model's name, as completions report it.
    getter name : String

    # The system message a chat starts with when it has none of its own.
    property system : String?

    @end_tokens : Array(Int32)

    # Returns an engine running *model*. Generation ends at any of *end_tokens*,
    # by default whichever of `<|endoftext|>` and `<|im_end|>` the tokeniser has.
    def initialize(model : M, @tokenizer : Tensomite::BPE, @name : String,
                   end_tokens : Array(Int32)? = nil, @system : String? = nil) forall M
      @decoder = ModelDecoder(M).new(model).as(Decoder)
      @end_tokens = end_tokens || ["<|endoftext|>", "<|im_end|>"].compact_map { |token| tokenizer.id?(token) }
      @lock = Mutex.new
    end

    # Returns an engine for the model in *dir*, named after the directory.
    #
    # A directory whose `config.json` has `model_type` `llama` holds a Llama-family
    # model, with `model.safetensors` and `tokenizer.json`; generation also ends
    # at its `eos_token_id`, and a chat's default system message is the one
    # `tokenizer_config.json`'s chat template adds, if any. Otherwise it holds a
    # GPT-2, with `model.safetensors`, `vocab.json` and `merges.txt`, whose
    # attention *heads* are, unless given, `n_head` in `config.json`, or 12.
    def self.load(dir : String, heads : Int32?, device : Tensomite::Device) : Engine
      return load_gguf(dir, device) if dir.ends_with?(".gguf")
      name = File.basename(File.expand_path(dir))
      if Server.model_type(dir) == "llama"
        tokenizer = Tensomite::BPE.from_json(File.join(dir, "tokenizer.json"))
        model = Tensomite::Llama.load(dir, device)
        ends = ["<|endoftext|>", "<|im_end|>"].compact_map { |token| tokenizer.id?(token) }
        ends |= Server.configured_eos(dir)
        return new(model, tokenizer, name, ends, Server.default_system(dir))
      end
      heads ||= Server.configured_heads(dir) || 12
      model = Tensomite::GPT2.load(File.join(dir, "model.safetensors"), heads: heads, device: device)
      tokenizer = Tensomite::BPE.from_files(File.join(dir, "vocab.json"), File.join(dir, "merges.txt"))
      new(model, tokenizer, name)
    end

    # Returns an engine for the Llama-family model in the GGUF file at *path*,
    # named after the file, with the tokeniser, end token and chat template
    # its metadata holds.
    def self.load_gguf(path : String, device : Tensomite::Device) : Engine
      file = Tensomite::GGUF.open(path)
      tokenizer = Tensomite::BPE.from_gguf(file)
      ends = ["<|endoftext|>", "<|im_end|>"].compact_map { |token| tokenizer.id?(token) }
      file.metadata["tokenizer.ggml.eos_token_id"]?.try { |eos| ends |= [eos.as_i] }
      system = file.metadata["tokenizer.chat_template"]?.try { |template| Server.template_system(template.as_s) }
      new(Tensomite::Llama.load(file, device), tokenizer, File.basename(path, ".gguf"), ends, system)
    end

    # Returns true when the tokeniser has ChatML's tokens, so the engine can chat.
    def chat? : Bool
      !@tokenizer.id?("<|im_start|>").nil? && !@tokenizer.id?("<|im_end|>").nil?
    end

    # Returns the token ids of *messages* in ChatML, with the default system
    # message first when they have none, then the start of the assistant's turn.
    #
    # Each message is `<|im_start|>role\ncontent<|im_end|>\n`, its role and
    # content encoded as plain text, so text that looks like a special token
    # cannot end a message early.
    def chat_ids(messages : Array(Message)) : Array(Int32)
      raise RequestError.new("this model's tokeniser has no ChatML tokens, so it cannot chat") unless chat?
      raise RequestError.new("`messages` is empty") if messages.empty?
      start, stop = @tokenizer.specials["<|im_start|>"], @tokenizer.specials["<|im_end|>"]
      if (system = @system) && messages.first.role != "system"
        messages = [Message.new("system", system)] + messages
      end
      newline = @tokenizer.encode("\n", special: false)
      ids = messages.flat_map do |message|
        [start] + @tokenizer.encode("#{message.role}\n#{message.content}", special: false) + [stop] + newline
      end
      ids + [start] + @tokenizer.encode("assistant\n", special: false)
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
      prompt = request.messages.try { |messages| chat_ids(messages) } || @tokenizer.encode(request.prompt)
      raise RequestError.new("the prompt is empty") if prompt.empty?
      context = @decoder.context
      if prompt.size >= context
        raise RequestError.new("the prompt's #{prompt.size} tokens leave no room in the context of #{context}")
      end
      limit = Math.min(request.max_tokens, context - prompt.size)
      sampler = Tensomite::Sampler.new(temperature: request.temperature, top_k: request.top_k, seed: request.seed)
      held = request.stop.max_of?(&.size).try { |size| size - 1 } || 0
      forward = @decoder.session(prompt.size + limit)
      logits = forward.call(prompt)
      bytes = IO::Memory.new
      text, sent, count, reason, cut_at_stop = "", 0, 0, "length", false
      while count < limit
        token = sampler.pick(logits.to_a(Float32))
        if @end_tokens.includes?(token)
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
        logits = forward.call([token]) if count < limit
      end
      # A character cut by the end of the completion becomes U+FFFD.
      text = String.new(bytes.to_slice).scrub unless cut_at_stop
      yield text[sent..] if text.size > sent
      Completion.new(text, prompt.size, count, reason)
    end
  end

  # Returns `n_head` from *dir*'s `config.json`, or `nil` when it has none.
  def self.configured_heads(dir : String) : Int32?
    config(dir).try(&.["n_head"]?).try(&.as_i?)
  end

  # Returns `model_type` from *dir*'s `config.json`, or `nil` when it has none.
  def self.model_type(dir : String) : String?
    config(dir).try(&.["model_type"]?).try(&.as_s?)
  end

  # Returns the ids of `eos_token_id` in *dir*'s `config.json`, one or several.
  def self.configured_eos(dir : String) : Array(Int32)
    case eos = config(dir).try(&.["eos_token_id"]?)
    when Nil then [] of Int32
    else          eos.as_a?.try(&.map(&.as_i)) || eos.as_i?.try { |id| [id] } || [] of Int32
    end
  end

  # Returns the system message the chat template in *dir*'s
  # `tokenizer_config.json` adds to a chat without one, or `nil`.
  def self.default_system(dir : String) : String?
    path = File.join(dir, "tokenizer_config.json")
    return unless File.exists?(path)
    template = JSON.parse(File.read(path))["chat_template"]?.try(&.as_s?) || return
    template_system(template)
  end

  # Returns the system message a chat template adds to a chat without one, or `nil`.
  def self.template_system(template : String) : String?
    template.match(/'<\|im_start\|>system\n(.*?)<\|im_end\|>/m).try(&.[1])
  end

  private def self.config(dir : String) : JSON::Any?
    path = File.join(dir, "config.json")
    JSON.parse(File.read(path)) if File.exists?(path)
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
