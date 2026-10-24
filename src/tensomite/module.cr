module Tensomite
  @@rng = Random.new(0)

  # The generator modules draw their initial weights from when given none.
  # Seeded, so a program builds the same weights on every run.
  def self.rng : Random
    @@rng
  end

  def self.rng=(@@rng : Random)
  end

  # A model's learnable value: a slot holding a tensor, which an optimizer replaces
  # at each step. The tensor itself never changes.
  class Parameter
    getter value : Tensor
    # The parameter's path in its model, such as `"hidden1.weight"`; set when the
    # model lists its parameters.
    getter name : String = ""
    # Whether weight decay applies: true for weight matrices, false for biases
    # and normalisation parameters.
    getter? decay : Bool

    def initialize(@value : Tensor, @decay : Bool = true)
    end

    # Replaces the value, carrying the parameter's name over to the new tensor.
    def value=(value : Tensor) : Tensor
      @value = name.empty? ? value : value.named(name)
    end

    # :nodoc:
    def name=(@name : String) : String
      @value = @value.named(name)
      name
    end

    def shape : Array(Int32)
      value.shape
    end
  end

  # How a module fills a new parameter.
  struct Init
    enum Kind
      Uniform
      Normal
      Constant
    end

    getter kind : Kind
    getter scale : Float64

    def initialize(@kind : Kind, @scale : Float64)
    end

    # Values drawn uniformly from `[-bound, bound]`.
    def self.uniform(bound : Number) : Init
      new(Kind::Uniform, bound.to_f64)
    end

    # Values drawn from a normal distribution with mean 0 and standard deviation *std*.
    def self.normal(std : Number) : Init
      new(Kind::Normal, std.to_f64)
    end

    # Every value *value*.
    def self.constant(value : Number) : Init
      new(Kind::Constant, value.to_f64)
    end

    # Returns a tensor of *shape* filled as this says, drawing from *rng* in
    # row-major order. Normal values use the Box–Muller transform, two draws each.
    def tensor(shape : Array(Int32), rng : Random, device : Device) : Tensor
      size = shape.product(1)
      data = case kind
             in .uniform?
               Array(Float32).new(size) { (rng.rand(-1.0..1.0) * scale).to_f32 }
             in .normal?
               Array(Float32).new(size) { (scale * Math.sqrt(-2 * Math.log(1 - rng.rand)) * Math.cos(2 * Math::PI * rng.rand)).to_f32 }
             in .constant?
               Array(Float32).new(size, scale.to_f32)
             end
      Tensor.from(data, shape, device)
    end
  end

  # A part of a model: parameters, and modules within it, combined by `forward`.
  #
  # A subclass declares its parts as instance variables of type `Parameter`,
  # `Module`, `Array(Parameter)`, `Array(Module)` or nilable versions of them;
  # they are found at compile time and named after the variables, so nothing is
  # registered by hand. It defines `forward` with whatever arguments it needs, and
  # is used through `#call`, which runs `forward` inside a scope named after the
  # module's place in its parent (see `Tensomite.scope`).
  #
  # ```
  # class Multiplier < Tensomite::Module
  #   def initialize
  #     @hidden = Linear.new(2, 64)
  #     @output = Linear.new(64, 1)
  #   end
  #
  #   def forward(x : Tensor) : Tensor
  #     @output.call(@hidden.call(x).gelu)
  #   end
  # end
  #
  # Multiplier.new.named_params.keys # => ["hidden.weight", "hidden.bias", "output.weight", "output.bias"]
  # ```
  abstract class Module
    @scope_name : String? = nil

    # Runs `forward` with *args*, inside this module's scope when it has a parent.
    def call(*args, **options)
      each_child do |name, child|
        child.scope_name = child_scope_name(name, child) if child.is_a?(Module)
      end
      if name = @scope_name
        Tensomite.scope(name) { forward(*args, **options) }
      else
        forward(*args, **options)
      end
    end

    # Returns every parameter, depth first in declaration order, each once.
    def params : Array(Parameter)
      named_params.values
    end

    # Returns every parameter by its path, such as `"blocks.0.attention.weight"`.
    # A parameter reached twice, as tied weights are, keeps its first path.
    def named_params : Hash(String, Parameter)
      result = {} of String => Parameter
      collect("", result)
      result
    end

    # Moves every parameter to *device*, and returns `self`.
    def to(device : Device) : self
      params.each { |param| param.value = param.value.to(device) }
      self
    end

    # Returns a short name for this kind of module, the class name in lower case;
    # `Sequential` names its steps this way.
    def kind : String
      self.class.name.split("::").last.downcase
    end

    # :nodoc:
    def scope_name=(@scope_name : String?) : String?
    end

    # :nodoc:
    def collect(prefix : String, result : Hash(String, Parameter)) : Nil
      each_child do |name, child|
        path = prefix.empty? ? name : "#{prefix}.#{name}"
        case child
        in Parameter
          next if result.each_value.any?(&.same?(child))
          child.name = path
          result[path] = child
        in Module
          child.collect(path, result)
        end
      end
    end

    # Returns the scope a child at *name* runs in: the name without an array index,
    # so that the members of an array share one scope name and fold in diagrams.
    protected def child_scope_name(name : String, child : Module) : String
      name.split('.').first
    end

    # Yields each parameter and child module with its name in this module: the
    # instance variable's name, and `name.i` for the members of an array.
    def each_child(& : String, Parameter | Module ->) : Nil
      {% for ivar in @type.instance_vars %}
        {% types = ivar.type.union? ? ivar.type.union_types.reject { |type| type == Nil } : [ivar.type] %}
        {% if types.size == 1 %}
          {% type = types[0] %}
          {% if type <= ::Tensomite::Parameter || type <= ::Tensomite::Module %}
            if part = @{{ ivar.id }}
              yield {{ ivar.name.stringify }}, part
            end
          {% elsif type.name(generic_args: false) == "Array" && (type.type_vars[0] <= ::Tensomite::Parameter || type.type_vars[0] <= ::Tensomite::Module) %}
            if parts = @{{ ivar.id }}
              parts.each_with_index { |part, i| yield "{{ ivar.name }}.#{i}", part }
            end
          {% end %}
        {% end %}
      {% end %}
    end
  end

  # A module that takes one tensor and returns one, so that `Sequential` can chain it.
  abstract class Layer < Module
    abstract def forward(x : Tensor) : Tensor
  end

  struct Tensor
    # Returns the flowchart `#to_mermaid` draws with *depth*, counting *model*'s parameters.
    def to_mermaid(model : Module, depth : Int32 = 1, max_nodes : Int32 = 200) : String
      to_mermaid(max_nodes: max_nodes, depth: depth, params: model.params.map(&.value))
    end
  end
end
