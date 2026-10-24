module Tensomite
  # Updates parameters from their gradients.
  #
  # Given a model's `Parameter`s when created, `#step(loss)` differentiates the
  # loss and replaces each parameter's value:
  #
  # ```
  # optimizer = AdamW.new(model.params, lr: 1e-3)
  # loss_value = optimizer.step(loss)
  # ```
  #
  # Created without them, `#step(params, grads)` takes tensors and returns new,
  # realised ones to pass to the next forward pass, since tensors are immutable.
  # Either way the optimizer keeps its own state, such as momentum, per param in
  # the order given.
  abstract class Optimizer
    # The number of steps taken.
    getter steps = 0

    # The parameters `#step(loss)` updates, when given.
    getter slots : Array(Parameter)? = nil

    # Differentiates *loss* with respect to the parameters given at creation,
    # replaces each one's value with its update, and returns the loss's value.
    # Raises `Error` for an optimizer created without parameters.
    def step(loss : Tensor) : Float32
      slots = @slots || raise Error.new("step(loss) needs an optimizer created with parameters")
      values = slots.map(&.value)
      grads = Tensomite.grad(loss, values)
      value = loss.to_a(Float32).first
      slots.zip(step(values, grads)) { |slot, updated| slot.value = updated }
      value
    end

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

    # Returns whether weight decay applies to param *index*: its `Parameter`'s
    # flag when the optimizer has parameters, and always otherwise.
    protected def decays?(index : Int32) : Bool
      (slots = @slots) ? slots[index].decay? : true
    end
  end

  # Stochastic gradient descent, with optional momentum:
  # `v = momentum * v + g`, then `p -= lr * v`.
  class SGD < Optimizer
    getter lr : Float64
    getter momentum : Float64

    @velocity = {} of Int32 => Tensor

    def initialize(@lr : Float64, @momentum : Float64 = 0.0)
    end

    # Creates an optimizer for *params*, for `#step(loss)`.
    def initialize(params : Array(Parameter), @lr : Float64, @momentum : Float64 = 0.0)
      @slots = params
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
  # 1. `p -= lr * weight_decay * p`, for parameters that decay
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

    # Creates an optimizer for *params*, for `#step(loss)`. Weight decay applies
    # only to the parameters whose `Parameter#decay?` is true.
    def initialize(params : Array(Parameter), @lr : Float64 = 1e-3, @beta1 : Float64 = 0.9, @beta2 : Float64 = 0.999,
                   @eps : Float64 = 1e-8, @weight_decay : Float64 = 0.01)
      @slots = params
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
      decayed = weight_decay == 0 || !decays?(index) ? param : param * (1 - lr * weight_decay)
      decayed - m_hat / (v_hat.sqrt + eps) * lr
    end
  end
end
