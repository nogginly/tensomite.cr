module Tensomite
  # `x · weight + bias`, for `x` of shape `[..., inputs]`.
  #
  # The weight is `[inputs, outputs]`, drawn from *init*: by default uniformly
  # from ±1/√inputs, as PyTorch does, so outputs start at about the size of
  # inputs. The bias, if any, starts at zero and is not decayed.
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
      product = x.matmul(weight.value)
      (b = bias) ? product + b.value : product
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
  # The queries, keys and values come from one `Linear` (`qkv`), whose output is
  # split into thirds and each third into *heads*; `proj` maps the heads'
  # joined outputs back to the width. Weights are drawn from *init*, and `proj`'s
  # from *proj_init*, which GPT-2 scales down by the depth.
  class CausalSelfAttention < Module
    getter heads : Int32
    getter qkv : Linear
    getter proj : Linear

    def initialize(width : Int32, @heads : Int32, bias : Bool = true,
                   init : Init = Init.normal(0.02), proj_init : Init = init,
                   rng : Random = Tensomite.rng, device : Device = CPU.default)
      raise ShapeError.new("width #{width} does not divide into #{heads} heads") unless width % heads == 0
      @qkv = Linear.new(width, 3 * width, bias: bias, init: init, rng: rng, device: device)
      @proj = Linear.new(width, width, bias: bias, init: proj_init, rng: rng, device: device)
    end

    # Returns the attention of *x*, `[batch * length, width]`, under *mask*, a
    # `[length, length]` mask such as `Tensor.causal_mask(length, length)`.
    #
    # 1. Projects *x* to queries, keys and values, each `[batch, heads, length, head width]`.
    # 2. Scores queries against keys, scaled by 1/√(head width), masked, and
    #    normalised with softmax; the weights then average the values.
    # 3. Rejoins the heads and projects back to the width.
    def forward(x : Tensor, mask : Tensor) : Tensor
      rows, width = x.shape
      length = mask.shape[0]
      batch, head_width = rows // length, width // heads
      joined = qkv.call(x)
      q, k, v = (0..2).map do |part|
        joined.slice(1, part * width, (part + 1) * width)
          .reshape([batch, length, heads, head_width]).permute([0, 2, 1, 3])
      end
      scores = q.matmul(k.transpose) * (1 / Math.sqrt(head_width)) + mask
      attended = scores.softmax.matmul(v).permute([0, 2, 1, 3]).reshape([batch * length, width])
      proj.call(attended)
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
