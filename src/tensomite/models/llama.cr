require "json"

module Tensomite
  # A Llama-family decoder, such as SmolLM2, loaded from a Hugging Face model
  # directory: its `config.json` and `model.safetensors`.
  #
  # ```
  # model = Llama.load("models/smollm2-135m-instruct")
  # cache = model.new_cache
  # logits = model.forward(ids, cache)       # [ids.size, vocab]
  # logits = model.forward([next_id], cache) # one more position, reusing the cache
  # ```
  #
  # It is a `Module` whose parts are named as in Hugging Face's weights file
  # (`model.embed_tokens`, `model.layers.0.self_attn.q_proj`, `model.norm`,
  # and `lm_head` unless the head shares the embeddings), so the file loads by
  # name. PyTorch stores each `Linear`'s weight transposed, so loading copies
  # them into tensomite's `[inputs, outputs]`.
  class Llama < Module
    # Sizes of the model, and the constants of its rotary embeddings and norms.
    record Config,
      layers : Int32, heads : Int32, kv_heads : Int32, width : Int32, hidden : Int32,
      head_width : Int32, context : Int32, vocab : Int32,
      rope_theta : Float64 = 10_000.0, eps : Float64 = 1e-5, tied : Bool = true do
      # Returns the configuration in a Hugging Face `config.json`.
      #
      # Raises `Error` for a configuration this model would run wrongly, such as
      # scaled rotary embeddings or biases in the MLP.
      def self.from_json(json : String) : Config
        config = JSON.parse(json)
        unless config["rope_scaling"]?.try(&.raw).nil?
          raise Error.new("rope_scaling #{config["rope_scaling"].to_json} is not supported")
        end
        rope = config["rope_parameters"]? || config
        unless rope["rope_type"]?.try(&.as_s).in?(nil, "default")
          raise Error.new("rope_type #{rope["rope_type"]} is not supported")
        end
        %w[attention_bias mlp_bias].each do |flag|
          raise Error.new("#{flag} is not supported") if config[flag]?.try(&.as_bool)
        end
        width, heads = config["hidden_size"].as_i, config["num_attention_heads"].as_i
        new(
          layers: config["num_hidden_layers"].as_i, heads: heads,
          kv_heads: config["num_key_value_heads"]?.try(&.as_i) || heads,
          width: width, hidden: config["intermediate_size"].as_i,
          head_width: config["head_dim"]?.try(&.as_i?) || width // heads,
          context: config["max_position_embeddings"].as_i, vocab: config["vocab_size"].as_i,
          rope_theta: rope["rope_theta"]?.try { |theta| theta.as_f? || theta.as_i.to_f } || 10_000.0,
          eps: config["rms_norm_eps"]?.try { |eps| eps.as_f? || eps.as_i.to_f } || 1e-5,
          tied: config["tie_word_embeddings"]?.try(&.as_bool) || false)
      end
    end

    # A transformer block: attention, then an MLP, each after an RMS norm and
    # added back to the residual stream.
    class Block < Module
      def initialize(config : Config, build : Builder)
        @input_layernorm = build.rmsnorm(config.width, config.eps)
        @self_attn = build.grouped_attention(config.width, config.heads, config.kv_heads, config.head_width, bias: false)
        @post_attention_layernorm = build.rmsnorm(config.width, config.eps)
        @mlp = build.swiglu(config.width, config.hidden)
      end

      def forward(x : Tensor, mask : Tensor, rope : RotaryEmbedding, cache : CausalSelfAttention::Cache? = nil) : Tensor
        x = x + @self_attn.call(@input_layernorm.call(x), mask, rope, cache)
        x + @mlp.call(@post_attention_layernorm.call(x))
      end
    end

    # The embeddings, blocks and final norm: Hugging Face's `model`.
    class Body < Module
      getter embed_tokens : Embedding
      getter layers : Array(Block)
      getter norm : RMSNorm

      def initialize(config : Config, build : Builder)
        @embed_tokens = build.embedding(config.vocab, config.width)
        @layers = Array.new(config.layers) { Block.new(config, build) }
        @norm = build.rmsnorm(config.width, config.eps)
      end
    end

    # Keys and values of every position run so far, one `CausalSelfAttention::Cache` per block.
    class Cache
      getter layers : Array(CausalSelfAttention::Cache)

      # Returns the number of positions held.
      getter length = 0

      # A cache of *count* layers, each holding up to *capacity* positions.
      def initialize(count : Int32, capacity : Int32? = nil)
        @layers = Array.new(count) { CausalSelfAttention::Cache.new(capacity) }
      end

      protected def advance(count : Int32) : Nil
        @length += count
      end
    end

    # Weights stored transposed in Hugging Face's file: every `Linear`'s.
    TRANSPOSED = /_proj\.weight$|^lm_head\.weight$/

    getter config : Config
    getter device : Device
    getter rope : RotaryEmbedding
    @model : Body
    @lm_head : Linear?

    # Returns the model in the Hugging Face directory *path*, from its
    # `config.json` and `model.safetensors`, with its weights on *device*.
    # `BF16` weights are widened to `F32`.
    def self.load(path : String, device : Device = CPU.default) : Llama
      config = Config.from_json(File.read(File.join(path, "config.json")))
      new(config, Builder.new(device: device, init: Init.placeholder))
        .load(File.join(path, "model.safetensors"), transposed: TRANSPOSED)
    end

    # Returns a model of *config*'s sizes, its parts made by *build*.
    def initialize(@config : Config, build : Builder)
      @device = build.device
      @model = Body.new(config, build)
      @lm_head = build.linear(config.width, config.vocab, bias: false) unless config.tied
      @rope = RotaryEmbedding.new(config.head_width, config.context, config.rope_theta, device)
    end

    # Returns an empty cache for this model, holding up to *capacity* positions
    # in place.
    def new_cache(capacity : Int32 = config.context) : Cache
      Cache.new(config.layers, capacity)
    end

    # Runs *ids* at the positions following those in *cache*, appends their keys and
    # values to *cache*, and returns realised logits of shape `[ids.size, vocab]`.
    #
    # 1. Embeds the tokens.
    # 2. Runs each block: attention then MLP, each after an RMS norm and added
    #    back to the residual stream, with queries and keys rotated by position.
    # 3. Applies the final norm and projects onto the vocabulary.
    def forward(ids : Array(Int32), cache : Cache) : Tensor
      result = logits(ids, cache).realize
      cache.advance(count: ids.size)
      result
    end

    # Returns lazy logits `[batch * length, vocab]` for *ids*, `[batch, length]`
    # token ids, each row its own sequence from position 0, without a cache.
    def forward(ids : Tensor) : Tensor
      batch, length = ids.shape
      if length > config.context
        raise ShapeError.new("#{length} positions exceed the context of #{config.context}")
      end
      x = Tensomite.scope("embedding") { @model.embed_tokens.call(ids).reshape([batch * length, config.width]) }
      mask = Tensor.causal_mask(length, length, device).named("causal mask")
      Tensomite.scope("layers") { @model.layers.each { |block| x = block.call(x, mask, rope) } }
      head(x)
    end

    # Returns *count* tokens generated after *prompt*, each chosen by *sampler* from
    # the logits of the last position, yielding each as it is chosen. Generation
    # stops early after a token in *stop*, which is then the last one returned.
    def generate(prompt : Array(Int32), count : Int32, sampler : Sampler = Sampler.greedy,
                 stop : Enumerable(Int32) = [] of Int32, & : Int32 ->) : Array(Int32)
      cache = new_cache(Math.min(config.context, prompt.size + count))
      logits = forward(prompt, cache)
      tokens = [] of Int32
      count.times do |i|
        token = sampler.pick(logits.slice(0, logits.shape[0] - 1, logits.shape[0]).to_a(Float32))
        tokens << token
        yield token
        break if stop.includes?(token)
        logits = forward([token], cache) if i < count - 1
      end
      tokens
    end

    # :ditto:
    def generate(prompt : Array(Int32), count : Int32, sampler : Sampler = Sampler.greedy,
                 stop : Enumerable(Int32) = [] of Int32) : Array(Int32)
      generate(prompt, count, sampler, stop) { }
    end

    # Returns the logits of *ids* after the positions in *cache*, unrealised, and
    # appends their keys and values to it; `#forward` advances it after realising.
    private def logits(ids : Array(Int32), cache : Cache) : Tensor
      count, past = ids.size, cache.length
      raise ShapeError.new("forward of no tokens") if count == 0
      if past + count > config.context
        raise ShapeError.new("#{past + count} positions exceed the context of #{config.context}")
      end
      x = Tensomite.scope("embedding") { @model.embed_tokens.call(Tensor.from(ids, [count], device).named("ids")) }
      mask = Tensor.causal_mask(count, past + count, device).named("causal mask")
      Tensomite.scope("layers") do
        @model.layers.each_with_index { |block, layer| x = block.call(x, mask, rope, cache.layers[layer]) }
      end
      head(x)
    end

    private def head(x : Tensor) : Tensor
      Tensomite.scope("head") do
        normed = @model.norm.call(x)
        if lm_head = @lm_head
          lm_head.call(normed)
        else
          normed.matmul(@model.embed_tokens.weight.value.transpose)
        end
      end
    end
  end
end
