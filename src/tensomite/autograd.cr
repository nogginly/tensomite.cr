module Tensomite
  # Returns the gradient of *loss*, a one-element tensor, with respect to each of
  # *params*, as lazy tensors of the params' shapes.
  #
  # The gradients are a graph built from the unrealised graph of *loss*, so
  # realising them runs the forward ops they need too. Realise *loss* and the
  # gradients together, or *loss* after calling `grad`: once a node is realised its
  # sources are gone, and `grad` raises `Error` when it meets one. Use
  # `Tensor#detach` to make an intermediate result a constant.
  #
  # A param the loss does not depend on gets a gradient of zeros.
  #
  # ```
  # w = Tensor.from([1.0_f32, 2.0_f32], [2])
  # loss = (w * w).sum
  # Tensomite.grad(loss, [w]).first.to_a(Float32) # => [2.0, 4.0]
  # ```
  def self.grad(loss : Tensor, params : Array(Tensor)) : Array(Tensor)
    Tensor.grad(loss, params)
  end

  struct Tensor
    # See `Tensomite.grad`.
    #
    # 1. Lists the nodes of the graph in topological order, stopping at leaves.
    # 2. Marks the nodes that lead to a param; only these receive gradients.
    # 3. Seeds the loss node with ones, then walks the order backwards. Each node's
    #    accumulated gradient, in the node's shape, is turned into one gradient per
    #    marked source by the op's chain rule, and added to that source's node. The
    #    nodes a chain rule builds are in the scope of the node it differentiates.
    # 4. Reads each param's gradient through the param's view of its node.
    def self.grad(loss : Tensor, params : Array(Tensor)) : Array(Tensor)
      raise ShapeError.new("grad needs a one-element loss, not #{loss.shape}") unless loss.size == 1
      raise DTypeError.new("grad needs a float loss, not #{loss.dtype}") unless loss.dtype.float?
      order = graph_order(loss.node)
      wanted = params.map(&.node).to_set
      needed = Set(Node).new
      order.each do |node|
        needed << node if wanted.includes?(node) || node.srcs.any? { |src| needed.includes?(src.node) }
      end
      grads = {} of Node => Tensor
      if needed.includes?(loss.node)
        accumulate(grads, loss, full(loss.shape, 1, loss.dtype, loss.device))
      end
      order.reverse_each do |node|
        next unless (g = grads[node]?) && node.op
        Scope.within(node.scope) do
          backward(node, g).each_with_index do |src_grad, i|
            src = node.srcs[i]
            accumulate(grads, src, src_grad) if src_grad && needed.includes?(src.node)
          end
        end
      end
      params.map do |param|
        g = grads[param.node]? || zeros(param.node.shape, param.dtype, param.device)
        through(g, param.view)
      end
    end

    # Returns the nodes *root* depends on, sources before the nodes that use them.
    private def self.graph_order(root : Node) : Array(Node)
      order = [] of Node
      seen = Set(Node).new
      stack = [{root, false}]
      while entry = stack.pop?
        node, expanded = entry
        if expanded
          order << node
        elsif seen.add?(node)
          if node.realized? && node.op
            raise Error.new("grad met a realised #{node.op} node, whose sources are gone; " \
                            "call grad before realising, or detach constants")
          end
          stack << {node, true}
          node.srcs.reverse_each { |src| stack << {src.node, false} }
        end
      end
      order
    end

    # Adds *g*, the gradient of *src* in src's shape, to the gradient of src's node.
    private def self.accumulate(grads : Hash(Node, Tensor), src : Tensor, g : Tensor) : Nil
      node = src.node
      g = src.view.contiguous? && src.size == node.size ? g.reshape(node.shape) : src.unview(g)
      grads[node] = (existing = grads[node]?) ? existing + g : g
    end

    # Returns *g*, in the shape of the node it was computed for, read through *view*.
    private def self.through(g : Tensor, view : View) : Tensor
      g = g.contiguous
      unless g.size == g.node.size
        g = Tensor.lazy(Node.new(Op::Copy, [g], g.shape, g.dtype, g.device))
      end
      Tensor.new(g.node, view)
    end

    # Returns, for each source of *node*, its gradient in the source's shape given
    # *g*, the gradient of *node*'s output; `nil` for sources without one. `a` is the
    # first source and `b` the last, the second operand of two-source ops.
    private def self.backward(node : Node, g : Tensor) : Array(Tensor?)
      srcs = node.srcs
      a = srcs.first
      b = srcs.last
      result = Tensor.lazy(node)
      case node.op
      in Op::Add then [g, g] of Tensor?
      in Op::Sub then [g, -g] of Tensor?
      in Op::Mul then [g * b, g * a] of Tensor?
      in Op::Div then [g / b, -(g * result) / b] of Tensor?
      in Op::Maximum
        mask = a.greater_equal(b)
        [g * mask, g * (-mask + 1)] of Tensor?
      in Op::Exp  then [g * result] of Tensor?
      in Op::Log  then [g / a] of Tensor?
      in Op::Tanh then [g * (-(result * result) + 1)] of Tensor?
      in Op::Sqrt then [g / (result * 2)] of Tensor?
      in Op::Sum  then [spread(g, a.shape, node.axis)] of Tensor?
      in Op::Max
        mask = a.greater_equal(spread(result, a.shape, node.axis))
        [spread(g, a.shape, node.axis) * mask / spread(mask.sum(node.axis), a.shape, node.axis)] of Tensor?
      in Op::Copy then [g] of Tensor?
      in Op::Cat
        axis = node.axis || raise Error.new("cat needs an axis")
        start = 0
        srcs.map do |src|
          stop = start + src.shape[axis]
          g.slice(axis, start, stop).tap { start = stop }.as(Tensor?)
        end
      in Op::Gather
        rows, width = a.shape
        [g.reshape([b.size, width]).scatter_add_rows(b.reshape([b.size]), rows), nil] of Tensor?
      in Op::MatMul
        [g.matmul(b.transpose), a.transpose.matmul(g)] of Tensor?
      in Op::ScatterAdd then [g.gather(b), nil] of Tensor?
      in Op::Unview
        [through(g, node.source_view || raise Error.new("unview needs a source view"))] of Tensor?
      in Op::Softmax
        # dx = y · (g - Σ g·y), per row.
        [result * (g - (g * result).sum(-1, keepdims: true))] of Tensor?
      in Op::LogSoftmax
        # dx = g - exp(y) · Σ g, per row.
        [g - result.exp * g.sum(-1, keepdims: true)] of Tensor?
      in Op::LayerNorm
        layernorm_backward(node, g)
      in Op::Kernel
        kernel = node.kernel || raise Error.new("kernel node without a kernel")
        srcs.map_with_index do |_, i|
          gradient = kernel.gradient(i) || raise Error.new("kernel #{kernel.name} has no gradient of its own")
          Tensor.kernel(gradient, [g] + srcs).as(Tensor?)
        end
      in Op::GreaterEqual
        [nil, nil] of Tensor?
      in Op::Write
        raise Error.new("write has no gradient")
      in Nil
        raise Error.new("leaf nodes have no sources")
      end
    end

    # Returns the gradients of a fused layer norm's input, weight and bias, from
    # the normalised input `x̂ = (x - mean) / σ` recomputed by primitives:
    #
    # - weight: `g · x̂`, and bias: `g`, each in the expanded shape, so that
    #   accumulating them sums over the rows;
    # - input: `(ĝ - mean(ĝ) - x̂ · mean(ĝ · x̂)) / σ`, where `ĝ = g · weight`.
    private def self.layernorm_backward(node : Node, g : Tensor) : Array(Tensor?)
      x, weight, _ = node.srcs
      epsilon = node.epsilon || raise Error.new("layernorm needs an epsilon")
      centred = x - x.mean(-1, keepdims: true)
      std = ((centred * centred).mean(-1, keepdims: true) + epsilon).sqrt
      normalised = centred / std
      scaled = g * weight
      dx = (scaled - scaled.mean(-1, keepdims: true) - normalised * (scaled * normalised).mean(-1, keepdims: true)) / std
      [dx, g * normalised, g] of Tensor?
    end

    # Returns *g*, the result of reducing over *axis* (all axes when `nil`), broadcast
    # back to the unreduced *shape*.
    private def self.spread(g : Tensor, shape : Array(Int32), axis : Int32?) : Tensor
      kept = axis ? shape.dup.tap { |dims| dims[axis] = 1 } : Array.new(shape.size, 1)
      g.reshape(kept).expand(shape)
    end
  end
end
