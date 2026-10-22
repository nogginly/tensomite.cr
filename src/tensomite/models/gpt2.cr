module Tensomite
  # The GPT-2 decoder, loaded from Hugging Face safetensors weights.
  #
  # ```
  # model = GPT2.load("model.safetensors")
  # cache = model.new_cache
  # logits = model.forward(ids, cache)       # [ids.size, vocab]
  # logits = model.forward([next_id], cache) # one more position, reusing the cache
  # ```
  class GPT2
    # Sizes of the model: transformer blocks, attention heads, embedding width,
    # longest context and vocabulary.
    record Config, layers : Int32, heads : Int32, width : Int32, context : Int32, vocab : Int32

    # The weights of one transformer block, in GPT-2's `[in, out]` layout.
    private record Block,
      ln1_w : Tensor, ln1_b : Tensor, attn_w : Tensor, attn_b : Tensor,
      proj_w : Tensor, proj_b : Tensor, ln2_w : Tensor, ln2_b : Tensor,
      fc_w : Tensor, fc_b : Tensor, out_w : Tensor, out_b : Tensor

    # Keys and values of every position run so far, per layer, each `[heads, length, head width]`.
    class Cache
      getter keys : Array(Tensor?)
      getter values : Array(Tensor?)

      # Returns the number of positions held.
      getter length = 0

      def initialize(layers : Int32)
        @keys = Array(Tensor?).new(layers, nil)
        @values = Array(Tensor?).new(layers, nil)
      end

      protected def advance(count : Int32) : Nil
        @length += count
      end
    end

    getter config : Config
    getter device : Device
    @wte : Tensor
    @wpe : Tensor
    @lnf_w : Tensor
    @lnf_b : Tensor
    @blocks : Array(Block)

    # Returns the model in the safetensors file at *path*, with *heads* attention
    # heads (12 for GPT-2 small), its weights on *device*.
    def self.load(path : String, heads : Int32 = 12, device : Device = CPU.default) : GPT2
      new(SafeTensors.open(path), heads, device)
    end

    # Returns the model from *weights*, named as by Hugging Face with or without the
    # `transformer.` prefix. Sizes other than *heads* come from the weight shapes.
    def initialize(weights : SafeTensors, heads : Int32, @device : Device)
      prefix = weights.has_key?("transformer.wte.weight") ? "transformer." : ""
      fetch = ->(name : String) { weights[prefix + name, device] }
      @wte = fetch.call("wte.weight")
      @wpe = fetch.call("wpe.weight")
      @lnf_w = fetch.call("ln_f.weight")
      @lnf_b = fetch.call("ln_f.bias")
      layers = (0..).index! { |i| !weights.has_key?("#{prefix}h.#{i}.ln_1.weight") }
      @blocks = Array(Block).new(layers) do |i|
        part = ->(name : String) { fetch.call("h.#{i}.#{name}") }
        Block.new(
          part.call("ln_1.weight"), part.call("ln_1.bias"),
          part.call("attn.c_attn.weight"), part.call("attn.c_attn.bias"),
          part.call("attn.c_proj.weight"), part.call("attn.c_proj.bias"),
          part.call("ln_2.weight"), part.call("ln_2.bias"),
          part.call("mlp.c_fc.weight"), part.call("mlp.c_fc.bias"),
          part.call("mlp.c_proj.weight"), part.call("mlp.c_proj.bias"))
      end
      vocab, width = @wte.shape
      raise ShapeError.new("width #{width} does not divide into #{heads} heads") unless width % heads == 0
      @config = Config.new(layers, heads, width, @wpe.shape[0], vocab)
    end

    # Returns an empty cache for this model.
    def new_cache : Cache
      Cache.new(config.layers)
    end

    # Runs *ids* at the positions following those in *cache*, appends their keys and
    # values to *cache*, and returns realised logits of shape `[ids.size, vocab]`.
    #
    # 1. Embeds tokens and positions.
    # 2. Runs each block: attention then MLP, each after a layer norm and added
    #    back to the residual stream.
    # 3. Applies the final layer norm and projects onto the token embeddings.
    def forward(ids : Array(Int32), cache : Cache) : Tensor
      count, past = ids.size, cache.length
      raise ShapeError.new("forward of no tokens") if count == 0
      if past + count > config.context
        raise ShapeError.new("#{past + count} positions exceed the context of #{config.context}")
      end
      x = @wte.gather(Tensor.from(ids, [count], device)) + @wpe.slice(0, past, past + count)
      mask = Tensor.causal_mask(count, past + count, device)
      @blocks.each_with_index do |block, layer|
        x = x + attention(block, x.layernorm(block.ln1_w, block.ln1_b), cache, layer, mask)
        hidden = x.layernorm(block.ln2_w, block.ln2_b).matmul(block.fc_w) + block.fc_b
        x = x + (hidden.gelu.matmul(block.out_w) + block.out_b)
      end
      logits = x.layernorm(@lnf_w, @lnf_b).matmul(@wte.transpose).realize
      cache.advance(count)
      logits
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

    # Returns causal self-attention over the cache and *h*, the normalised input.
    #
    # 1. Projects *h* to queries, keys and values, each split into heads.
    # 2. Appends the new keys and values to the cache along the position axis.
    # 3. Scores queries against all keys, scaled by 1/√(head width), masked, and
    #    normalised with softmax; the weights then average the values.
    # 4. Rejoins the heads and projects back to the model width.
    private def attention(block : Block, h : Tensor, cache : Cache, layer : Int32, mask : Tensor) : Tensor
      count, width, heads = h.shape[0], config.width, config.heads
      head_width = width // heads
      qkv = h.matmul(block.attn_w) + block.attn_b
      q, k, v = (0..2).map do |part|
        qkv.slice(1, part * width, (part + 1) * width).reshape([count, heads, head_width]).permute([1, 0, 2])
      end
      keys = cache.keys[layer].try { |past| Tensor.cat([past, k], 1) } || k
      values = cache.values[layer].try { |past| Tensor.cat([past, v], 1) } || v
      cache.keys[layer], cache.values[layer] = keys, values
      scores = q.matmul(keys.transpose) * (1 / Math.sqrt(head_width)) + mask
      attended = scores.softmax.matmul(values)
      attended.permute([1, 0, 2]).reshape([count, width]).matmul(block.proj_w) + block.proj_b
    end
  end
end
