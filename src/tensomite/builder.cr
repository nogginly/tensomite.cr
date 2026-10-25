module Tensomite
  # Makes built-in modules that share a generator, a device and defaults, so a
  # model states them once rather than at every part.
  #
  # Each method makes the module of the same name, filling in `rng:` and
  # `device:`, and `bias:` and `init:` when not given. A model takes a builder as
  # an argument and passes it on to the modules it contains:
  #
  # ```
  # class Multiplier < Tensomite::Module
  #   def initialize(build : Builder = Builder.new)
  #     @mlp = Sequential.build(build) do
  #       linear 2, 64
  #       gelu
  #       linear 64, 1
  #     end
  #   end
  #
  #   def forward(x : Tensor) : Tensor
  #     @mlp.call(x)
  #   end
  # end
  #
  # Multiplier.new(Builder.new(rng: Random.new(1), device: BLAS.default))
  # ```
  struct Builder
    getter rng : Random
    getter device : Device
    # Whether `linear` and `attention` add biases unless told otherwise.
    getter? bias : Bool
    # How weights are drawn unless told otherwise; `nil` leaves each module's own default.
    getter init : Init?

    def initialize(@rng : Random = Tensomite.rng, @device : Device = CPU.default,
                   @bias : Bool = true, @init : Init? = nil)
    end

    # Returns a builder with the given settings changed.
    def with(rng : Random = @rng, device : Device = @device, bias : Bool = @bias, init : Init? = @init) : Builder
      Builder.new(rng, device, bias, init)
    end

    def linear(inputs : Int32, outputs : Int32, bias : Bool = @bias, init : Init? = @init) : Linear
      Linear.new(inputs, outputs, bias: bias, init: init || Init.uniform(1 / Math.sqrt(inputs)), rng: rng, device: device)
    end

    def embedding(count : Int32, width : Int32, init : Init? = @init) : Embedding
      Embedding.new(count, width, init: init || Init.normal(0.02), rng: rng, device: device)
    end

    def layernorm(width : Int32, eps : Float64 = 1e-5) : LayerNorm
      LayerNorm.new(width, eps, device: device)
    end

    def attention(width : Int32, heads : Int32, bias : Bool = @bias, init : Init? = @init, proj_init : Init? = nil) : CausalSelfAttention
      weights = init || Init.normal(0.02)
      CausalSelfAttention.new(width, heads, bias: bias, init: weights, proj_init: proj_init || weights, rng: rng, device: device)
    end

    # Returns a parameter of *shape* drawn from *init*.
    def parameter(shape : Array(Int32), init : Init? = @init, decay : Bool = true) : Parameter
      Parameter.new((init || Init.normal(0.02)).tensor(shape, rng, device), decay: decay)
    end

    def gelu : Activation
      Activation.gelu
    end

    def tanh : Activation
      Activation.tanh
    end
  end

  class Sequential
    # Returns a `Sequential` of the layers the block adds. Inside the block,
    # `linear`, `layernorm`, `gelu` and `tanh` add a layer made by *build*, and
    # `add` adds any layer.
    #
    # ```
    # @mlp = Sequential.build(build) do
    #   linear 2, 64
    #   gelu
    #   linear 64, 1
    # end
    # ```
    #
    # This is a class method, not a `Builder` method, so that an instance
    # variable assigned from it gets its type: Crystal infers types from class
    # methods called with a block, but not from instance methods.
    def self.build(build : Builder = Builder.new, &) : Sequential
      steps = SequentialBuilder.new(build)
      with steps yield
      new(steps.layers)
    end
  end

  # The receiver inside `Sequential.build`'s block.
  class SequentialBuilder
    getter layers = [] of Layer

    def initialize(@build : Builder)
    end

    def add(layer : Layer) : Nil
      layers << layer
    end

    {% for name in %w[linear layernorm gelu tanh] %}
      def {{ name.id }}(*args, **options) : Nil
        add @build.{{ name.id }}(*args, **options)
      end
    {% end %}
  end
end
