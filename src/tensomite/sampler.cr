module Tensomite
  # Chooses a token from a row of logits.
  #
  # Greedy sampling takes the highest logit, the first on ties. Otherwise logits are
  # divided by the temperature, cut to the *top_k* highest when it is positive, and
  # sampled from their softmax with a seeded generator, so a seed repeats a run.
  #
  # ```
  # Sampler.greedy.pick(logits)
  # Sampler.new(temperature: 0.8, top_k: 40, seed: 42).pick(logits)
  # ```
  class Sampler
    getter temperature : Float64
    getter top_k : Int32

    # Returns a sampler that always picks the highest logit.
    def self.greedy : Sampler
      new(temperature: 0.0)
    end

    def initialize(@temperature : Float64 = 1.0, @top_k : Int32 = 0, seed : Int32 = 0)
      raise ArgumentError.new("temperature must not be negative") if @temperature < 0
      @random = Random::PCG32.new(seed.to_u64)
    end

    # Returns the index of the chosen token.
    def pick(logits : Array(Float32)) : Int32
      return logits.index!(logits.max) if temperature == 0
      candidates = (0...logits.size).to_a
      candidates = candidates.sort_by { |i| -logits[i] }.first(top_k) if 0 < top_k < logits.size
      top = candidates.max_of { |i| logits[i] }.to_f64
      weights = candidates.map { |i| Math.exp((logits[i] - top) / temperature) }
      draw = @random.rand * weights.sum
      candidates.each_with_index do |token, rank|
        draw -= weights[rank]
        return token if draw < 0
      end
      candidates.last
    end
  end
end
