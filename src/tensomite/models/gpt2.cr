module Tensomite
  # The GPT-2 decoder, loaded from Hugging Face safetensors weights.
  #
  # ```
  # model = GPT2.load("model.safetensors")
  # cache = model.new_cache
  # logits = model.forward(ids, cache)       # [ids.size, vocab]
  # logits = model.forward([next_id], cache) # one more position, reusing the cache
  # ```
  #
  # It is a `Module` whose parts are named as in Hugging Face's weights file
  # (`wte`, `wpe`, `h.0.attn.c_attn`, `ln_f`), so the file loads by name.
  class GPT2 < Module
    # Sizes of the model: transformer blocks, attention heads, embedding width,
    # longest context and vocabulary.
    record Config, layers : Int32, heads : Int32, width : Int32, context : Int32, vocab : Int32

    # GPT-2's MLP: a `Linear` to four times the width, `gelu`, and a `Linear` back.
    class MLP < Layer
      def initialize(width : Int32, build : Builder)
        @c_fc = build.linear(width, 4 * width)
        @c_proj = build.linear(4 * width, width)
      end

      def forward(x : Tensor) : Tensor
        @c_proj.call(@c_fc.call(x).gelu)
      end
    end

    # A transformer block: attention, then an MLP, each after a layer norm and
    # added back to the residual stream.
    class Block < Module
      def initialize(width : Int32, heads : Int32, build : Builder)
        @ln_1 = build.layernorm(width)
        @attn = build.attention(width, heads)
        @ln_2 = build.layernorm(width)
        @mlp = MLP.new(width, build)
      end

      def forward(x : Tensor, mask : Tensor, cache : CausalSelfAttention::Cache) : Tensor
        x = x + @attn.call(@ln_1.call(x), mask, cache)
        x + @mlp.call(@ln_2.call(x))
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

    # Tensors in Hugging Face's file that are not weights: each block's stored causal mask.
    IGNORED = /\.attn\.(masked_)?bias$/

    getter config : Config
    getter device : Device
    @wte : Embedding
    @wpe : Embedding
    @h : Array(Block)
    @ln_f : LayerNorm

    # Returns the model in the safetensors file at *path*, with *heads* attention
    # heads (12 for GPT-2 small), its weights on *device*.
    def self.load(path : String, heads : Int32 = 12, device : Device = CPU.default) : GPT2
      new(SafeTensors.open(path), heads, device)
    end

    # Returns the model from *weights*, named as by Hugging Face with or without the
    # `transformer.` prefix. Sizes other than *heads* come from the weight shapes.
    def self.new(weights : SafeTensors, heads : Int32, device : Device) : GPT2
      prefix = weights.has_key?("transformer.wte.weight") ? "transformer." : ""
      vocab, width = weights.entries[prefix + "wte.weight"].shape
      context = weights.entries[prefix + "wpe.weight"].shape[0]
      layers = (0..).index! { |i| !weights.has_key?("#{prefix}h.#{i}.ln_1.weight") }
      config = Config.new(layers, heads, width, context, vocab)
      new(config, Builder.new(device: device, init: Init.placeholder)).load(weights, prefix: prefix, ignore: IGNORED)
    end

    # Returns a model of *config*'s sizes, its parts made by *build*.
    def initialize(@config : Config, build : Builder)
      @device = build.device
      @wte = build.embedding(config.vocab, config.width)
      @wpe = build.embedding(config.context, config.width)
      @h = Array.new(config.layers) { Block.new(config.width, config.heads, build) }
      @ln_f = build.layernorm(config.width)
    end

    # Returns an empty cache for this model, holding up to its context of
    # positions in place.
    def new_cache : Cache
      Cache.new(config.layers, config.context)
    end

    # Runs *ids* at the positions following those in *cache*, appends their keys and
    # values to *cache*, and returns realised logits of shape `[ids.size, vocab]`.
    #
    # 1. Embeds tokens and positions.
    # 2. Runs each block: attention then MLP, each after a layer norm and added
    #    back to the residual stream.
    # 3. Applies the final layer norm and projects onto the token embeddings.
    def forward(ids : Array(Int32), cache : Cache) : Tensor
      result = logits(ids, cache).realize
      cache.advance(count: ids.size)
      result
    end

    # Returns the unrealised logits of *ids* from an empty cache, for diagrams
    # (`Tensor#to_mermaid`). The graph is scoped as `embedding`, then `h` for each
    # block, holding `attn` and `mlp`, then `head`.
    def graph(ids : Array(Int32)) : Tensor
      logits(ids, new_cache)
    end

    # Returns *count* tokens generated after *prompt*, each chosen by *sampler* from
    # the logits of the last position, yielding each as it is chosen.
    def generate(prompt : Array(Int32), count : Int32, sampler : Sampler = Sampler.greedy, & : Int32 ->) : Array(Int32)
      cache = new_cache
      logits = forward(prompt, cache)
      Array(Int32).new(count) do |i|
        token = sampler.pick(logits.slice(0, logits.shape[0] - 1, logits.shape[0]).to_a(Float32))
        yield token
        logits = forward([token], cache) if i < count - 1
        token
      end
    end

    # :ditto:
    def generate(prompt : Array(Int32), count : Int32, sampler : Sampler = Sampler.greedy) : Array(Int32)
      generate(prompt, count, sampler) { }
    end

    # Returns the logits of *ids* after the positions in *cache*, unrealised, and
    # appends their keys and values to it; `#forward` advances it after realising.
    private def logits(ids : Array(Int32), cache : Cache) : Tensor
      count, past = ids.size, cache.length
      raise ShapeError.new("forward of no tokens") if count == 0
      if past + count > config.context
        raise ShapeError.new("#{past + count} positions exceed the context of #{config.context}")
      end
      x = Tensomite.scope("embedding") do
        @wte.call(Tensor.from(ids, [count], device).named("ids")) + @wpe.weight.value.slice(0, past, past + count)
      end
      mask = Tensor.causal_mask(count, past + count, device).named("causal mask")
      @h.each_with_index { |block, layer| x = block.call(x, mask, cache.layers[layer]) }
      Tensomite.scope("head") { @ln_f.call(x).matmul(@wte.weight.value.transpose) }
    end
  end
end
