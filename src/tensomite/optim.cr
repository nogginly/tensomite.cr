module Tensomite
  # Updates parameters from their gradients.
  #
  # Tensors are immutable, so `#step` returns new, realised parameters rather than
  # changing the old ones; pass them to the next forward pass. The optimizer keeps
  # its own state, such as momentum, for the params in the order given.
  #
  # ```
  # optimizer = AdamW.new(lr: 1e-3)
  # params = optimizer.step(params, Tensomite.grad(loss, params))
  # ```
  abstract class Optimizer
    # The number of steps taken.
    getter steps = 0

    # Returns *params* moved against *grads*, one gradient per param, realised and
    # detached from the graphs that made them.
    def step(params : Array(Tensor), grads : Array(Tensor)) : Array(Tensor)
      raise ShapeError.new("#{grads.size} gradients for #{params.size} params") unless grads.size == params.size
      params.zip(grads).each do |(param, grad)|
        raise ShapeError.new("gradient #{grad.shape} for param #{param.shape}") unless param.shape == grad.shape
      end
      @steps += 1
      params.map_with_index { |param, i| update(i, param, grads[i]).detach }
    end

    # Returns the new value of param *index*, lazily.
    protected abstract def update(index : Int32, param : Tensor, grad : Tensor) : Tensor
  end

  # Stochastic gradient descent, with optional momentum:
  # `v = momentum * v + g`, then `p -= lr * v`.
  class SGD < Optimizer
    getter lr : Float64
    getter momentum : Float64

    @velocity = {} of Int32 => Tensor

    def initialize(@lr : Float64, @momentum : Float64 = 0.0)
    end

    protected def update(index : Int32, param : Tensor, grad : Tensor) : Tensor
      if momentum != 0
        grad = @velocity[index] = (prev = @velocity[index]?) ? (prev * momentum + grad).detach : grad.detach
      end
      param - grad * lr
    end
  end

  # Adam with decoupled weight decay (Loshchilov and Hutter, 2019):
  #
  # 1. `p -= lr * weight_decay * p`
  # 2. `m = β₁m + (1 - β₁)g` and `v = β₂v + (1 - β₂)g²`
  # 3. `p -= lr * m̂ / (√v̂ + eps)`, where `m̂` and `v̂` divide out the bias of
  #    starting from zero: `m̂ = m / (1 - β₁ᵗ)` and `v̂ = v / (1 - β₂ᵗ)`.
  class AdamW < Optimizer
    property lr : Float64
    getter beta1 : Float64
    getter beta2 : Float64
    getter eps : Float64
    getter weight_decay : Float64

    @m = {} of Int32 => Tensor
    @v = {} of Int32 => Tensor

    def initialize(@lr : Float64 = 1e-3, @beta1 : Float64 = 0.9, @beta2 : Float64 = 0.999,
                   @eps : Float64 = 1e-8, @weight_decay : Float64 = 0.01)
    end

    protected def update(index : Int32, param : Tensor, grad : Tensor) : Tensor
      m = grad * (1 - beta1)
      v = grad * grad * (1 - beta2)
      if prev = @m[index]?
        m = prev * beta1 + m
        v = @v[index] * beta2 + v
      end
      m = @m[index] = m.detach
      v = @v[index] = v.detach
      m_hat = m / (1 - beta1 ** steps)
      v_hat = v / (1 - beta2 ** steps)
      decayed = weight_decay == 0 ? param : param * (1 - lr * weight_decay)
      decayed - m_hat / (v_hat.sqrt + eps) * lr
    end
  end
end
