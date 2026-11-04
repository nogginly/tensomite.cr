module Tensomite
  # Dropout for one forward pass in training: each `#call` zeroes a fresh random
  # share *rate* of a tensor's elements, with a seed of its own drawn from
  # *seed* and how many times it has been called. A forward pass that calls it
  # in the same order therefore makes the same masks whenever it is given the
  # same seed, on every device.
  #
  # ```
  # dropout = Dropout.new(0.1, seed: 1337_u64 + step)
  # loss = model.call(inputs, dropout: dropout).cross_entropy(targets)
  # ```
  class Dropout
    getter rate : Float64

    def initialize(@rate : Float64, @seed : UInt64)
      raise ArgumentError.new("dropout rate #{rate} is not in [0, 1)") unless 0 <= rate < 1
      @uses = 0_u64
    end

    # Returns *x* with dropout applied, under the next seed.
    def call(x : Tensor) : Tensor
      @uses += 1
      x.dropout(rate, Dropout.mix(@seed &+ @uses &* 0x9e3779b97f4a7c15_u64))
    end

    # Returns *x* mixed by SplitMix64's finaliser, so that nearby inputs give
    # unrelated seeds.
    def self.mix(x : UInt64) : UInt64
      x = (x ^ (x >> 30)) &* 0xbf58476d1ce4e5b9_u64
      x = (x ^ (x >> 27)) &* 0x94d049bb133111eb_u64
      x ^ (x >> 31)
    end
  end
end
