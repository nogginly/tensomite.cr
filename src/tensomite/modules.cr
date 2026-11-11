module Tensomite
  # `x · weight + bias`, for `x` of shape `[..., inputs]`.
  #
  # The weight is `[inputs, outputs]`, drawn from *init*: by default uniformly
  # from ±1/√inputs, as PyTorch does, so outputs start at about the size of
  # inputs. The bias, if any, starts at zero and is not decayed.
  #
  # A quantised weight, from `#quantize` or a GGUF file, is held as GGUF holds
  # it, `[outputs, inputs]` in a block format, and applied with `Tensor#linear`.
  class Linear < Layer
    getter weight : Parameter
    getter bias : Parameter?

    def initialize(inputs : Int32, outputs : Int32, bias : Bool = true,
                   init : Init = Init.uniform(1 / Math.sqrt(inputs)),
                   rng : Random = Tensomite.rng, device : Device = CPU.default)
      @weight = Parameter.new(init.tensor([inputs, outputs], rng, device))
      @bias = Parameter.new(Tensor.zeros([outputs], device: device), decay: false) if bias
    end

    def forward(x : Tensor) : Tensor
      value = weight.value
      value.dtype.block? ? x.linear(value, bias.try(&.value)) : x.matmul(value, bias.try(&.value))
    end

    # Joins the weights of *parts*, `Linear`s that read the same input, into one,
    # so that one product computes all their outputs, and returns it; each
    # part's weight becomes a view of its share, so nothing is held twice.
    # Returns `nil`, and changes nothing, when the parts have biases or weights
    # of different dtypes.
    #
    # Quantised weights, `[outputs, inputs]`, are joined by rows, and `F32`
    # ones, `[inputs, outputs]`, by columns. Each output is computed as the
    # part's own product computes it, so the results keep their bits on every
    # device whose products do not depend on their shape.
    def self.merge(parts : Array(Linear)) : Tensor?
      values = parts.map(&.weight.value)
      dtype = values.first.dtype
      return if parts.any?(&.bias) || values.any? { |value| value.dtype != dtype }
      if dtype.block?
        merged = Tensor.cat_rows(values)
        axis = 0
      else
        merged = Tensor.cat(values, 1).detach
        axis = 1
      end
      offset = 0
      parts.each do |part|
        size = part.weight.value.shape[axis]
        part.weight.value = merged.slice(axis, offset, offset + size)
        offset += size
      end
      merged
    end

    # Returns true when every one of *parts* still holds its share of *merged*,
    # as `.merge` left it, so that one product of *merged* stands for theirs.
    def self.merged?(parts : Array(Linear), merged : Tensor) : Bool
      parts.all?(&.weight.value.node.same?(merged.node))
    end

    # Returns `x` times a weight joined by `.merge`.
    def self.apply(x : Tensor, merged : Tensor) : Tensor
      merged.dtype.block? ? x.linear(merged) : x.matmul(merged)
    end

    # Replaces the weight with its transpose quantised to *dtype*, a block
    # format, so that each output's inputs are whole blocks. The inputs must be
    # a multiple of 32.
    def quantize(dtype : DType) : Nil
      value = weight.value
      return if value.dtype == dtype
      raise DTypeError.new("#{value.dtype} weights cannot be quantised again") if value.dtype.block?
      weight.value = value.transpose.contiguous.quantize(dtype)
    end
  end

  # Rows of a `[count, width]` table selected by `I32` ids: a token's id gives
  # its vector. The table is drawn from *init*, by default a normal distribution
  # with standard deviation 0.02, as GPT-2's.
  class Embedding < Layer
    getter weight : Parameter

    def initialize(count : Int32, width : Int32, init : Init = Init.normal(0.02),
                   rng : Random = Tensomite.rng, device : Device = CPU.default)
      @weight = Parameter.new(init.tensor([count, width], rng, device))
    end

    # Returns `[*ids.shape, width]` for ids of any shape.
    def forward(x : Tensor) : Tensor
      weight.value.gather(x)
    end

    # Replaces the table with itself quantised to *dtype*, a block format; the
    # width must be a multiple of 32. Lookups then dequantise the rows they read.
    def quantize(dtype : DType) : Nil
      return if weight.value.dtype == dtype
      raise DTypeError.new("#{weight.value.dtype} tables cannot be quantised again") if weight.value.dtype.block?
      weight.value = weight.value.quantize(dtype)
    end
  end

  # Layer normalisation of the last axis (see `Tensor#layernorm`), with a weight
  # starting at one and a bias at zero, neither decayed.
  class LayerNorm < Layer
    getter weight : Parameter
    getter bias : Parameter
    getter eps : Float64

    def initialize(width : Int32, @eps : Float64 = 1e-5, device : Device = CPU.default)
      @weight = Parameter.new(Tensor.full([width], 1, device: device), decay: false)
      @bias = Parameter.new(Tensor.zeros([width], device: device), decay: false)
    end

    def forward(x : Tensor) : Tensor
      x.layernorm(weight.value, bias.value, eps)
    end
  end

  # Multi-head causal self-attention over rows of `[batch * length, width]`, as
  # in GPT-2: each position attends to itself and the positions before it.
  #
  # The queries, keys and values come from one `Linear` (`c_attn`), whose output
  # is split into thirds and each third into *heads*; `c_proj` maps the heads'
  # joined outputs back to the width. The names are GPT-2's, so its weights load
  # by name. Weights are drawn from *init*, and `c_proj`'s from *proj_init*,
  # which GPT-2 scales down by the depth.
  class CausalSelfAttention < Module
    # The keys and values of the positions run so far, each `[batch, heads,
    # positions, head width]`, so that later positions can attend to them
    # without recomputing them.
    #
    # With a *capacity*, the cache holds buffers of that many positions, made
    # on the first append, and each append writes its positions into them in
    # place (`Tensor#write`). Without one, each append joins the new positions
    # to a copy of the old (`Tensor.cat`).
    class Cache
      getter capacity : Int32?

      # Returns the number of positions held.
      getter length = 0

      @keys : Tensor? = nil
      @values : Tensor? = nil

      def initialize(@capacity : Int32? = nil)
      end

      # Returns the keys held, or `nil` before the first append.
      def keys : Tensor?
        @keys.try(&.slice(2, 0, length))
      end

      # Returns the values held, or `nil` before the first append.
      def values : Tensor?
        @values.try(&.slice(2, 0, length))
      end

      # :nodoc:
      def append(keys : Tensor, values : Tensor) : {Tensor, Tensor}
        if capacity = @capacity
          stop = length + keys.shape[2]
          raise ShapeError.new("the cache holds #{capacity} positions, not #{stop}") if stop > capacity
          @keys = (@keys || Cache.empty(keys, capacity)).write(keys, 2, length)
          @values = (@values || Cache.empty(values, capacity)).write(values, 2, length)
        else
          @keys = @keys.try { |past| Tensor.cat([past, keys], 2) } || keys
          @values = @values.try { |past| Tensor.cat([past, values], 2) } || values
        end
        @length += keys.shape[2]
        {self.keys.as(Tensor), self.values.as(Tensor)}
      end

      # Returns a realised tensor shaped as *like*, but with *capacity* positions,
      # whose contents are undefined until written.
      protected def self.empty(like : Tensor, capacity : Int32) : Tensor
        shape = like.shape.dup
        shape[2] = capacity
        buffer = like.device.alloc(like.dtype.bytes_for(shape.product(1)))
        Tensor.new(Node.new(buffer, shape, like.dtype, like.device), View.contiguous(shape))
      end
    end

    getter heads : Int32
    getter c_attn : Linear
    getter c_proj : Linear

    def initialize(width : Int32, @heads : Int32, bias : Bool = true,
                   init : Init = Init.normal(0.02), proj_init : Init = init,
                   rng : Random = Tensomite.rng, device : Device = CPU.default)
      raise ShapeError.new("width #{width} does not divide into #{heads} heads") unless width % heads == 0
      @c_attn = Linear.new(width, 3 * width, bias: bias, init: init, rng: rng, device: device)
      @c_proj = Linear.new(width, width, bias: bias, init: proj_init, rng: rng, device: device)
    end

    # Returns the attention of *x*, `[batch * length, width]`, under *mask*,
    # `[length, positions]`, such as `Tensor.causal_mask(length, length)`.
    #
    # With a *cache*, the new keys and values are appended to it and *x*'s
    # positions attend to every position it holds, so *mask* covers them all:
    # `Tensor.causal_mask(length, past + length)`.
    #
    # 1. Projects *x* to queries, keys and values, each `[batch, heads, length, head width]`.
    # 2. Appends the keys and values to *cache*, if given.
    # 3. Scores queries against keys, scaled by 1/√(head width), masked, and
    #    normalised with softmax; the weights then average the values.
    # 4. Rejoins the heads and projects back to the width.
    #
    # With *dropout*, as in training, it applies to the attention weights and to
    # the projected output, as GPT-2's does.
    def forward(x : Tensor, mask : Tensor, cache : Cache? = nil, dropout : Dropout? = nil) : Tensor
      rows, width = x.shape
      length = mask.shape[0]
      batch, head_width = rows // length, width // heads
      joined = c_attn.call(x)
      q, k, v = (0..2).map do |part|
        joined.slice(1, part * width, (part + 1) * width)
          .reshape([batch, length, heads, head_width]).permute([0, 2, 1, 3])
      end
      k, v = cache.append(k, v) if cache
      weights = q.matmul(k.transpose).softmax(scale: 1 / Math.sqrt(head_width), mask: mask)
      weights = dropout.call(weights) if dropout
      attended = weights.matmul(v).permute([0, 2, 1, 3]).reshape([batch * length, width])
      output = c_proj.call(attended)
      dropout ? dropout.call(output) : output
    end
  end

  # Root-mean-square normalisation of the last axis (see `Tensor#rmsnorm`), with
  # a weight starting at one and not decayed, as in Llama.
  class RMSNorm < Layer
    getter weight : Parameter
    getter eps : Float64

    def initialize(width : Int32, @eps : Float64 = 1e-5, device : Device = CPU.default)
      @weight = Parameter.new(Tensor.full([width], 1, device: device), decay: false)
    end

    def forward(x : Tensor) : Tensor
      x.rmsnorm(weight.value, eps)
    end
  end

  # Rotary position embeddings, as in Llama: each head's features are taken as
  # pairs `(i, i + width/2)`, and the pair at position `p` is rotated by the
  # angle `p · θ^(-2i/width)`.
  #
  # The cosines and sines of every position up to *context* are computed once,
  # in `Float32` as PyTorch computes them, and kept on *device*.
  #
  # ```
  # rope = RotaryEmbedding.new(64, context: 2048, theta: 10_000.0)
  # q = rope.call(q, start: past) # q is [batch, heads, length, 64]
  # ```
  class RotaryEmbedding
    getter width : Int32
    getter context : Int32
    getter theta : Float64
    @cos : Tensor
    @sin : Tensor

    def initialize(@width : Int32, @context : Int32, @theta : Float64 = 10_000.0, device : Device = CPU.default)
      raise ShapeError.new("rotary embeddings need an even width, not #{width}") unless width.even?
      half = width // 2
      frequencies = Array(Float32).new(half) { |i| 1.0_f32 / (theta.to_f32 ** ((2 * i).to_f32 / width)) }
      cos = Array(Float32).new(context * width) { |idx| p, i = idx.divmod(width); Math.cos(p.to_f32 * frequencies[i % half]) }
      sin = Array(Float32).new(context * width) { |idx| p, i = idx.divmod(width); Math.sin(p.to_f32 * frequencies[i % half]) }
      @cos = Tensor.from(cos, [context, width], device).named("rope cos")
      @sin = Tensor.from(sin, [context, width], device).named("rope sin")
    end

    # Returns *x*, `[..., length, width]`, its rows rotated as positions
    # `start...start + length`: `x · cos + rotate_half(x) · sin`.
    def call(x : Tensor, start : Int32 = 0) : Tensor
      length = x.shape[-2]
      unless x.shape[-1] == width && start + length <= context
        raise ShapeError.new("rotary embeddings of width #{width} and context #{context} cannot rotate #{x.shape} from #{start}")
      end
      x.rope(@cos.slice(0, start, start + length), @sin.slice(0, start, start + length))
    end
  end

  # Causal self-attention with grouped queries and rotary embeddings, as in
  # Llama: *heads* query heads share *kv_heads* key and value heads, each shared
  # by `heads / kv_heads` consecutive query heads.
  #
  # Its parts are named as in Hugging Face's Llama (`q_proj`, `k_proj`,
  # `v_proj`, `o_proj`). The keys and values held in a
  # `CausalSelfAttention::Cache` have *kv_heads* heads.
  class GroupedQueryAttention < Module
    getter heads : Int32
    getter kv_heads : Int32
    getter q_proj : Linear
    getter k_proj : Linear
    getter v_proj : Linear
    getter o_proj : Linear
    @qkv : Tensor? = nil

    def initialize(width : Int32, @heads : Int32, @kv_heads : Int32, head_width : Int32 = width // heads,
                   bias : Bool = false, init : Init = Init.normal(0.02),
                   rng : Random = Tensomite.rng, device : Device = CPU.default)
      raise ShapeError.new("#{heads} query heads do not group into #{kv_heads} key and value heads") unless kv_heads > 0 && heads % kv_heads == 0
      @q_proj = Linear.new(width, heads * head_width, bias: bias, init: init, rng: rng, device: device)
      @k_proj = Linear.new(width, kv_heads * head_width, bias: bias, init: init, rng: rng, device: device)
      @v_proj = Linear.new(width, kv_heads * head_width, bias: bias, init: init, rng: rng, device: device)
      @o_proj = Linear.new(heads * head_width, width, bias: bias, init: init, rng: rng, device: device)
    end

    # Returns the attention of *x*, `[batch * length, width]`, under *mask*,
    # `[length, positions]`, with *rope* rotating queries and keys from the
    # first position not yet in *cache*.
    #
    # 1. Projects *x* to queries `[batch, heads, length, head width]`, and keys
    #    and values `[batch, kv_heads, length, head width]`.
    # 2. Rotates queries and keys by position.
    # 3. Appends the keys and values to *cache*, if given.
    # 4. Stacks each group's query heads as rows, `[batch, kv_heads, group *
    #    length, head width]`, so that one batched matmul scores them all
    #    against their shared keys, with *mask* repeated for each head.
    # 5. Normalises the scores with softmax and averages the values, as one
    #    op on a GPU while decoding (see `Tensor.attention`), and projects the
    #    heads' joined outputs back to the width.
    def forward(x : Tensor, mask : Tensor, rope : RotaryEmbedding, cache : CausalSelfAttention::Cache? = nil) : Tensor
      rows = x.shape[0]
      length = mask.shape[0]
      batch, head_width, group = rows // length, rope.width, heads // kv_heads
      start = cache ? cache.length : 0
      q, k, v = project(x)
      q = split(q, batch, length, heads, head_width)
      k = split(k, batch, length, kv_heads, head_width)
      v = split(v, batch, length, kv_heads, head_width)
      q, k = rope.call(q, start), rope.call(k, start)
      k, v = cache.append(k, v) if cache
      grouped = q.reshape([batch, kv_heads, group * length, head_width])
      masks = group == 1 ? mask : mask.expand([group] + mask.shape).reshape([group * length, mask.shape[1]])
      attended = Tensor.attention(grouped, k, v, scale: 1 / Math.sqrt(head_width), mask: masks).reshape([batch, heads, length, head_width])
      o_proj.call(attended.permute([0, 2, 1, 3]).reshape([rows, heads * head_width]))
    end

    # Joins the query, key and value projections into one product (see
    # `Linear.merge`), which `#forward` uses while they hold their shares of it.
    # Returns true when they could be joined.
    def merge : Bool
      !(@qkv = Linear.merge([q_proj, k_proj, v_proj])).nil?
    end

    # Returns the queries, keys and values of *x*: from one product of the
    # joined projections while `#merge`'s join holds, else from three.
    private def project(x : Tensor) : {Tensor, Tensor, Tensor}
      parts = [q_proj, k_proj, v_proj]
      if (joined = @qkv) && Linear.merged?(parts, joined)
        all = Linear.apply(x, joined)
        sizes = parts.map { |part| part.weight.value.dtype.block? ? part.weight.value.shape[0] : part.weight.value.shape[1] }
        {all.slice(1, 0, sizes[0]), all.slice(1, sizes[0], sizes[0] + sizes[1]), all.slice(1, sizes[0] + sizes[1], sizes.sum)}
      else
        {q_proj.call(x), k_proj.call(x), v_proj.call(x)}
      end
    end

    private def split(x : Tensor, batch : Int32, length : Int32, count : Int32, head_width : Int32) : Tensor
      x.reshape([batch, length, count, head_width]).permute([0, 2, 1, 3])
    end
  end

  # Llama's MLP: `down_proj(silu(gate_proj(x)) · up_proj(x))`, without biases,
  # its parts named as in Hugging Face's weights.
  class SwiGLU < Layer
    getter gate_proj : Linear
    getter up_proj : Linear
    getter down_proj : Linear
    @gate_up : Tensor? = nil

    def initialize(width : Int32, hidden : Int32, init : Init = Init.normal(0.02),
                   rng : Random = Tensomite.rng, device : Device = CPU.default)
      @gate_proj = Linear.new(width, hidden, bias: false, init: init, rng: rng, device: device)
      @up_proj = Linear.new(width, hidden, bias: false, init: init, rng: rng, device: device)
      @down_proj = Linear.new(hidden, width, bias: false, init: init, rng: rng, device: device)
    end

    def forward(x : Tensor) : Tensor
      if (joined = @gate_up) && Linear.merged?([gate_proj, up_proj], joined)
        both = Linear.apply(x, joined)
        hidden = both.shape[-1] // 2
        return down_proj.call(both.slice(-1, 0, hidden).swiglu(both.slice(-1, hidden, 2 * hidden)))
      end
      down_proj.call(gate_proj.call(x).swiglu(up_proj.call(x)))
    end

    # Joins the gate and up projections into one product (see `Linear.merge`),
    # which `#forward` uses while they hold their shares of it. Returns true
    # when they could be joined.
    def merge : Bool
      !(@gate_up = Linear.merge([gate_proj, up_proj])).nil?
    end
  end

  # An elementwise function with no parameters, as a layer, so that `Sequential`
  # can hold it.
  class Activation < Layer
    enum Function
      Gelu
      Tanh
    end

    getter function : Function

    def initialize(@function : Function)
    end

    def self.gelu : Activation
      new(Function::Gelu)
    end

    def self.tanh : Activation
      new(Function::Tanh)
    end

    def forward(x : Tensor) : Tensor
      case function
      in .gelu? then x.gelu
      in .tanh? then x.tanh
      end
    end

    # The function's name, such as `gelu`.
    def kind : String
      function.to_s.downcase
    end
  end

  # Layers applied in turn, each to the previous one's output.
  #
  # Each step runs in a scope named for its kind (`linear`, `gelu`, ...), and its
  # parameters are named by position: `steps.0.weight`.
  #
  # ```
  # mlp = Sequential.new(Linear.new(2, 64), Activation.gelu, Linear.new(64, 1))
  # ```
  class Sequential < Layer
    getter steps : Array(Layer)

    def initialize(*steps : Layer)
      @steps = steps.to_a.map(&.as(Layer))
    end

    def initialize(@steps : Array(Layer))
    end

    def forward(x : Tensor) : Tensor
      steps.reduce(x) { |input, step| step.call(input) }
    end

    protected def child_scope_name(name : String, child : Module) : String
      child.kind
    end
  end
end
