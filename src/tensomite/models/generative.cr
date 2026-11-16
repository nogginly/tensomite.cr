module Tensomite
  # What a decoder that generates text shares: a cache of every position run
  # so far, a forward pass through it, and generation. A model includes it
  # and provides `config.layers`, `config.context` and a private
  # `logits(ids : Array(Int32), cache : Cache, last : Bool = false) : Tensor`,
  # which returns the unrealised logits of *ids* after the positions in
  # *cache*, or with *last* those of the last of them only, and appends their
  # keys and values to it; and a private `packed_logits(appends :
  # Array(Array(Int32)), caches : Array(Cache), last : Bool) : Tensor`, which
  # does the same for several sequences at once, their new positions packed
  # one after another.
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

      # Empties the cache, keeping its buffers for the positions appended next.
      def reset : Nil
        @layers.each(&.reset)
        @length = 0
      end

      protected def advance(count : Int32) : Nil
        @length += count
      end
    end

    # A `Cache` for each of several rows, each a sequence of its own length, for
    # `#forward` and `#forward_last` over many sequences at once. A row's
    # buffers are made on its first append, and kept when it is `#reset`.
    #
    # ```
    # cache = model.new_batch_cache(rows: 64, capacity: 2048)
    # logits = model.forward_last(appends, cache) # appends[row] may be empty
    # cache.reset(3)                              # row 3 starts a new sequence
    # ```
    class BatchCache
      # Returns the number of rows.
      getter size : Int32

      getter capacity : Int32

      def initialize(@size : Int32, @layers : Int32, @capacity : Int32)
        @rows = Array(Cache?).new(size, nil)
      end

      # Returns the number of positions row *row* holds.
      def length(row : Int32) : Int32
        @rows[row].try(&.length) || 0
      end

      # Empties row *row*, keeping its buffers.
      def reset(row : Int32) : Nil
        @rows[row].try(&.reset)
      end

      # Returns row *row*'s cache, made on first use.
      def [](row : Int32) : Cache
        @rows[row] ||= Cache.new(@layers, @capacity)
      end
    end

    # Returns an empty cache for this model, holding up to *capacity* positions
    # in place, by default its context.
    def new_cache(capacity : Int32 = config.context) : Cache
      Cache.new(config.layers, capacity)
    end

    # Returns an empty cache of *rows* rows, each holding up to *capacity*
    # positions in place, by default this model's context.
    def new_batch_cache(rows : Int32, capacity : Int32 = config.context) : BatchCache
      BatchCache.new(rows, config.layers, capacity)
    end

    # Runs each row's *appends* at the positions following those its row of
    # *cache* holds, appends their keys and values to it, and returns realised
    # logits of every new position, `[total, vocab]`, the rows' one after
    # another. A row with no new tokens is left as it is.
    #
    # Each row's logits have the bits `#forward` gives that sequence alone, on
    # a shape-invariant device: the products run once on every row's new
    # positions, and attention runs per row, on the row's own cache.
    def forward(appends : Array(Array(Int32)), cache : BatchCache) : Tensor
      packed(appends, cache, last: false)
    end

    # Runs *appends* as the batched `#forward` does, and returns realised logits
    # of each row's last new position, `[rows with new tokens, vocab]`, in row
    # order, with the bits `#forward_last` gives each sequence alone.
    def forward_last(appends : Array(Array(Int32)), cache : BatchCache) : Tensor
      packed(appends, cache, last: true)
    end

    private def packed(appends : Array(Array(Int32)), cache : BatchCache, last : Bool) : Tensor
      raise ShapeError.new("#{appends.size} appends for #{cache.size} rows") unless appends.size == cache.size
      rows = (0...appends.size).reject { |row| appends[row].empty? }
      raise ShapeError.new("forward of no tokens") if rows.empty?
      caches = rows.map { |row| cache[row] }
      ids = rows.map { |row| appends[row] }
      ids.zip(caches) do |row_ids, row_cache|
        stop = row_cache.length + row_ids.size
        raise ShapeError.new("#{stop} positions exceed the context of #{config.context}") if stop > config.context
      end
      result = packed_logits(ids, caches, last).realize
      ids.zip(caches) { |row_ids, row_cache| row_cache.advance(count: row_ids.size) }
      result
    end

    # Returns, for layer *layer*, each packed row's segment: its rows of the
    # pack, its causal *masks* entry and its cache.
    private def segments(ids : Array(Array(Int32)), caches : Array(Cache), masks : Array(Tensor), layer : Int32) : Array(CausalSelfAttention::Segment)
      start = 0
      ids.map_with_index do |row_ids, i|
        segment = CausalSelfAttention::Segment.new(start, row_ids.size, masks[i], caches[i].layers[layer])
        start += row_ids.size
        segment
      end
    end

    # Returns each packed row's causal mask over its cached and new positions.
    private def packed_masks(ids : Array(Array(Int32)), caches : Array(Cache)) : Array(Tensor)
      ids.zip(caches).map { |row_ids, row_cache| Tensor.causal_mask(row_ids.size, row_cache.length + row_ids.size, device) }
    end

    # Returns the rows of *x*, packed rows' positions, that hold each row's last.
    private def last_rows(x : Tensor, ids : Array(Array(Int32))) : Tensor
      return x.slice(0, x.shape[0] - 1, x.shape[0]) if ids.size == 1
      ends = [] of Int32
      ids.reduce(0) { |start, row_ids| ends << start + row_ids.size - 1; start + row_ids.size }
      x.gather(Tensor.from(ends, [ends.size], device))
    end

    # Runs *ids* at the positions following those in *cache*, appends their keys and
    # values to *cache*, and returns realised logits of shape `[ids.size, vocab]`.
    def forward(ids : Array(Int32), cache : Cache) : Tensor
      result = logits(ids, cache).realize
      cache.advance(count: ids.size)
      result
    end

    # Runs *ids* as `#forward` does, and returns realised logits of the last
    # position only, `[1, vocab]`: the final norm and the head run on one row,
    # however many *ids* there are. Each row of the head depends only on its
    # own input row, so on a shape-invariant device the row has the bits of
    # `#forward`'s last.
    def forward_last(ids : Array(Int32), cache : Cache) : Tensor
      result = logits(ids, cache, last: true).realize
      cache.advance(count: ids.size)
      result
    end

    # Returns *count* tokens generated after *prompt*, each chosen by *sampler* from
    # the logits of the last position, yielding each as it is chosen. Generation
    # stops early after a token in *stop*, which is then the last one returned.
    def generate(prompt : Array(Int32), count : Int32, sampler : Sampler = Sampler.greedy,
                 stop : Enumerable(Int32) = [] of Int32, & : Int32 ->) : Array(Int32)
      cache = new_cache(Math.min(config.context, prompt.size + count))
      logits = forward_last(prompt, cache)
      tokens = [] of Int32
      count.times do |i|
        token = sampler.pick(logits.to_a(Float32))
        tokens << token
        yield token
        break if stop.includes?(token)
        logits = forward_last([token], cache) if i < count - 1
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
