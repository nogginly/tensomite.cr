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
      width = weight.shape[1]
      weight.value.gather(x.reshape([x.size])).reshape(x.shape + [width])
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
