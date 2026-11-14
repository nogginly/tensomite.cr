require "json"

module Tensomite
  # A Llama-family decoder, such as SmolLM2, loaded from a Hugging Face model
  # directory (its `config.json` and `model.safetensors`) or from a GGUF file.
  #
  # ```
  # model = Llama.load("models/smollm2-135m-instruct")
  # model = Llama.load("models/smollm2-135m-instruct-f16.gguf")
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

    # Returns the configuration in a GGUF file's `llama.*` metadata. The head
    # is tied unless the file holds `output.weight`.
    def self.config(file : GGUF) : Config
      meta = file.metadata
      unless (arch = meta["general.architecture"]?.try(&.as_s)) == "llama"
        raise Error.new("#{file.path} holds a #{arch} model, not a Llama")
      end
      key = ->(name : String) { meta["llama.#{name}"]? || raise Error.new("#{file.path} lacks llama.#{name}") }
      if meta.has_key?("llama.rope.scaling.type") && meta["llama.rope.scaling.type"].as_s != "none"
        raise Error.new("#{file.path}: rope scaling #{meta["llama.rope.scaling.type"]} is not supported")
      end
      width, heads = key.call("embedding_length").as_i, key.call("attention.head_count").as_i
      Config.new(
        layers: key.call("block_count").as_i, heads: heads,
        kv_heads: meta["llama.attention.head_count_kv"]?.try(&.as_i) || heads,
        width: width, hidden: key.call("feed_forward_length").as_i,
        head_width: meta["llama.rope.dimension_count"]?.try(&.as_i) || width // heads,
        context: key.call("context_length").as_i,
        vocab: meta["llama.vocab_size"]?.try(&.as_i) || file.entries["token_embd.weight"].shape[0],
        rope_theta: meta["llama.rope.freq_base"]?.try(&.as_f) || 10_000.0,
        eps: key.call("attention.layer_norm_rms_epsilon").as_f,
        tied: !file.has_key?("output.weight"))
    end

    # The names of GGUF's tensors, by the names of this model's parameters,
    # within each block and outside them.
    GGUF_BLOCK = {
      "input_layernorm.weight"          => "attn_norm.weight",
      "post_attention_layernorm.weight" => "ffn_norm.weight",
      "self_attn.q_proj.weight"         => "attn_q.weight",
      "self_attn.k_proj.weight"         => "attn_k.weight",
      "self_attn.v_proj.weight"         => "attn_v.weight",
      "self_attn.o_proj.weight"         => "attn_output.weight",
      "mlp.gate_proj.weight"            => "ffn_gate.weight",
      "mlp.up_proj.weight"              => "ffn_up.weight",
      "mlp.down_proj.weight"            => "ffn_down.weight",
    }
    GGUF_TOP = {"model.embed_tokens.weight" => "token_embd.weight", "model.norm.weight" => "output_norm.weight", "lm_head.weight" => "output.weight"}

    # Returns the GGUF name of parameter *name*.
    def self.gguf_name(name : String) : String
      if match = name.match(/^model\.layers\.(\d+)\.(.+)$/)
        "blk.#{match[1]}.#{GGUF_BLOCK[match[2]]}"
      else
        GGUF_TOP[name]
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

      # Joins the projections that read the same input: queries, keys and
      # values, and the MLP's gate and up (see `Linear.merge`).
      def merge : Nil
        @self_attn.merge
        @mlp.merge
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

    include Generative

    # The cache `Generative` keeps: see `Generative::Cache`.
    alias Cache = Generative::Cache

    # Weights stored transposed in Hugging Face's file: every `Linear`'s.
    TRANSPOSED = /_proj\.weight$|^lm_head\.weight$/

    getter config : Config
    getter device : Device
    getter rope : RotaryEmbedding
    @model : Body
    @lm_head : Linear?

    # Returns the model at *path*, with its weights on *device*: a GGUF file
    # when *path* ends in `.gguf`, else a Hugging Face directory, from its
    # `config.json` and `model.safetensors`. `BF16` and `F16` weights are
    # widened to `F32`.
    def self.load(path : String, device : Device = CPU.default) : Llama
      return load(GGUF.open(path), device) if path.ends_with?(".gguf")
      config = Config.from_json(File.read(File.join(path, "config.json")))
      new(config, Builder.new(device: device, init: Init.placeholder))
        .load(File.join(path, "model.safetensors"), transposed: TRANSPOSED)
        .tap(&.merge_projections)
    end

    # Returns the model in *file*, a GGUF file, with its weights on *device*.
    #
    # 1. Reads the configuration from the metadata.
    # 2. Reads each parameter's tensor by its GGUF name, raising for a missing
    #    one or for any tensor left over.
    # 3. Undoes llama.cpp's reordering of each head's query and key rows,
    #    which pairs features `(2i, 2i + 1)` for its own rotary embeddings,
    #    back to Hugging Face's `(i, i + width/2)`.
    # 4. Transposes each `F32` `Linear` weight, stored `[outputs, inputs]`, into
    #    tensomite's `[inputs, outputs]`. Quantised weights stay as they are
    #    stored, which is the layout `Linear` applies them in.
    def self.load(file : GGUF, device : Device = CPU.default) : Llama
      config = config(file)
      model = new(config, Builder.new(device: device, init: Init.placeholder))
      used = Set(String).new
      model.named_params.each do |name, param|
        key = gguf_name(name)
        raise Error.new("#{file.path} lacks #{key}") unless file.has_key?(key)
        used << key
        value = file[key, device]
        if TRANSPOSED.matches?(name)
          heads = name.ends_with?("q_proj.weight") ? config.heads : name.ends_with?("k_proj.weight") ? config.kv_heads : nil
          value = unpermute(value, heads) if heads
          value = value.transpose.contiguous.detach unless value.dtype.block?
        end
        expected = value.dtype.block? && TRANSPOSED.matches?(name) ? param.shape.reverse : param.shape
        raise ShapeError.new("#{key}: #{file.path} holds #{value.shape}, the model #{expected}") unless value.shape == expected
        param.value = value
      end
      extra = file.names.reject { |key| used.includes?(key) }
      raise Error.new("#{file.path} has #{extra.join(", ")}, which the model lacks") unless extra.empty?
      model.merge_projections
      model
    end

    # Joins, in every block, the projections that read the same input, so that
    # each block runs four products instead of seven (see `Linear.merge`). A
    # loaded model is already joined; a model whose weights are replaced, as in
    # training, runs its products separately until joined again.
    def merge_projections : self
      @model.layers.each(&.merge)
      self
    end

    # Quantises as `Module#quantize` does, then joins the projections again.
    def quantize(dtype : DType, embeddings : Bool = false, except : Regex? = nil) : self
      super
      merge_projections
    end

    # Returns *weight*, `[heads × head width, inputs]`, with each head's rows
    # reordered from llama.cpp's rotary layout, in which features `2i` and
    # `2i + 1` are paired, to Hugging Face's, in which `i` and `i + width/2` are.
    protected def self.unpermute(weight : Tensor, heads : Int32) : Tensor
      rows, inputs = weight.shape
      half = rows // heads // 2
      unless weight.dtype.block?
        return weight.reshape([heads, half, 2, inputs]).permute([0, 2, 1, 3]).reshape([rows, inputs])
      end
      order = [] of Int32
      heads.times { |head| 2.times { |pair| half.times { |i| order << head * 2 * half + 2 * i + pair } } }
      weight.take_rows(order)
    end

    # Returns a model of *config*'s sizes, its parts made by *build*.
    def initialize(@config : Config, build : Builder)
      @device = build.device
      @model = Body.new(config, build)
      @lm_head = build.linear(config.width, config.vocab, bias: false) unless config.tied
      @rope = RotaryEmbedding.new(config.head_width, config.context, config.rope_theta, device)
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

    # Returns the logits of *ids* after the positions in *cache*, unrealised, and
    # appends their keys and values to it; `#forward` advances it after realising.
    #
    # 1. Embeds the tokens.
    # 2. Runs each block: attention then MLP, each after an RMS norm and added
    #    back to the residual stream, with queries and keys rotated by position.
    # 3. Applies the final norm and projects onto the vocabulary.
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
          normed.linear(@model.embed_tokens.weight.value)
        end
      end
    end
  end
end
