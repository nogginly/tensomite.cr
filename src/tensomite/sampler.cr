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
    #
    # 1. Greedy: the first highest logit.
    # 2. Otherwise the candidates are every token, in index order, or with
    #    *top_k* the *top_k* highest, kept in one pass by a heap and then sorted
    #    highest first; equal logits keep the lower index first.
    # 3. Each candidate's weight is `exp((logit - highest) / temperature)`,
    #    summed in candidate order, and one draw from the generator walks the
    #    candidates in that order.
    def pick(logits : Array(Float32)) : Int32
      return logits.index!(logits.max) if temperature == 0
      return pick_among(top(logits, top_k), logits) if 0 < top_k < logits.size
      highest = logits.max.to_f64
      @weights.clear
      total = 0.0
      logits.each do |logit|
        weight = Math.exp((logit - highest) / temperature)
        @weights << weight
        total += weight
      end
      draw = @random.rand * total
      @weights.each_with_index do |weight, token|
        draw -= weight
        return token if draw < 0
      end
      logits.size - 1
    end

    @weights = [] of Float64
    @heap = [] of Int32

    private def pick_among(candidates : Array(Int32), logits : Array(Float32)) : Int32
      highest = logits[candidates.first].to_f64
      @weights.clear
      total = 0.0
      candidates.each do |token|
        weight = Math.exp((logits[token] - highest) / temperature)
        @weights << weight
        total += weight
      end
      draw = @random.rand * total
      candidates.each_with_index do |token, rank|
        draw -= @weights[rank]
        return token if draw < 0
      end
      candidates.last
    end

    # Returns the indices of the *count* highest logits, highest first, equal
    # logits by lower index: a min-heap of the best so far, whose root is the
    # one a better logit replaces.
    private def top(logits : Array(Float32), count : Int32) : Array(Int32)
      heap = @heap
      heap.clear
      logits.each_index do |token|
        if heap.size < count
          heap << token
          sift_up(heap, heap.size - 1, logits)
        elsif better?(token, heap[0], logits)
          heap[0] = token
          sift_down(heap, 0, logits)
        end
      end
      heap.sort { |a, b| better?(a, b, logits) ? -1 : (better?(b, a, logits) ? 1 : 0) }
    end

    private def better?(a : Int32, b : Int32, logits : Array(Float32)) : Bool
      logits[a] > logits[b] || (logits[a] == logits[b] && a < b)
    end

    private def sift_up(heap : Array(Int32), at : Int32, logits : Array(Float32)) : Nil
      while at > 0
        parent = (at - 1) // 2
        break unless better?(heap[parent], heap[at], logits)
        heap.swap(parent, at)
        at = parent
      end
    end

    private def sift_down(heap : Array(Int32), at : Int32, logits : Array(Float32)) : Nil
      loop do
        worst = at
        {2 * at + 1, 2 * at + 2}.each do |child|
          worst = child if child < heap.size && better?(heap[worst], heap[child], logits)
        end
        break if worst == at
        heap.swap(worst, at)
        at = worst
      end
    end
  end
end
