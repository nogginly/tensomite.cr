module Tensomite
  # What a decoder that generates text shares: a cache of every position run
  # so far, a forward pass through it, and generation. A model includes it
  # and provides `config.layers`, `config.context` and a private
  # `logits(ids : Array(Int32), cache : Cache) : Tensor`, which returns the
  # unrealised logits of *ids* after the positions in *cache* and appends
  # their keys and values to it.
  #
  # ```
  # cache = model.new_cache
  # logits = model.forward(bpe.encode("Hello"), cache) # [5, vocab]
  # model.generate(bpe.encode("Hello"), 12, stop: [eos])
  # ```
  module Generative
    # Keys and values of every position run so far, one
    # `CausalSelfAttention::Cache` per block.
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

    # Returns an empty cache for this model, holding up to *capacity* positions
    # in place, by default its context.
    def new_cache(capacity : Int32 = config.context) : Cache
      Cache.new(config.layers, capacity)
    end

    # Runs *ids* at the positions following those in *cache*, appends their keys and
    # values to *cache*, and returns realised logits of shape `[ids.size, vocab]`.
    def forward(ids : Array(Int32), cache : Cache) : Tensor
      result = logits(ids, cache).realize
      cache.advance(count: ids.size)
      result
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
  end
end
