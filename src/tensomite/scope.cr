module Tensomite
  # Runs the block inside a new scope named *name*, nested in the current one, and
  # returns the block's value. Every graph node built inside it records the scope,
  # so a diagram can group a model's nodes into its parts (see `Tensor#to_mermaid`).
  #
  # Each call opens a new instance: a loop that calls `scope("block")` four times
  # makes four sibling scopes, which diagrams fold into one when their graphs match.
  #
  # ```
  # y = Tensomite.scope("mlp") { (x.matmul(w1)).gelu.matmul(w2) }
  # y.node.scope.to_s # => "mlp"
  # ```
  def self.scope(name : String, &)
    Scope.within(Scope.new(name, Scope.current)) { yield }
  end

  # A named region of model code; see `Tensomite.scope`.
  #
  # The current scope is per process, not per fiber: build graphs from one fiber at a time.
  class Scope
    getter name : String
    getter parent : Scope?

    @@current : Scope? = nil

    def initialize(@name : String, @parent : Scope?)
    end

    # Returns the scope new nodes are built in, or `nil` outside every scope.
    def self.current : Scope?
      @@current
    end

    # Runs the block with *scope* as the current scope, then restores the previous one.
    def self.within(scope : Scope?, &)
      previous = @@current
      @@current = scope
      begin
        yield
      ensure
        @@current = previous
      end
    end

    # Returns the scopes from the outermost down to this one.
    def path : Array(Scope)
      scopes = [self]
      while scope = scopes.last.parent
        scopes << scope
      end
      scopes.reverse!
    end

    # Writes the names from the outermost scope down, joined by `/`.
    def to_s(io : IO) : Nil
      path.join(io, '/') { |scope, stream| stream << scope.name }
    end
  end
end
