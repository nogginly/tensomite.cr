module Tensomite
  # Freed device memory kept for reuse, keyed by size in bytes.
  #
  # Devices give memory back when the collector finalizes a buffer, and take it
  # before allocating anew, so a loop that builds the same graph every step, such
  # as training or decoding, stops allocating once it is warm.
  #
  # Finalizers can run in the middle of an allocation, including one made by the
  # pool itself. While the pool is changing it accepts nothing, and `#give`
  # returns false so that the caller frees the memory instead.
  class BufferPool(T)
    # The most bytes the pool holds; memory beyond it is left to be freed.
    getter limit : Int64

    # Bytes held now.
    getter held = 0_i64

    # Allocations served from the pool, and allocations it could not serve.
    getter hits = 0_i64
    getter misses = 0_i64

    @free = {} of Int32 => Array(T)
    @busy = false

    def initialize(@limit : Int64)
    end

    # Returns memory of exactly *bytesize* bytes, or `nil` when none is free.
    def take(bytesize : Int32) : T?
      if @busy || !(memory = @free[bytesize]?.try(&.pop?))
        @misses += 1
        return
      end
      @held -= bytesize
      @hits += 1
      memory
    end

    # Keeps *memory* of *bytesize* bytes and returns true, or returns false when
    # the pool is busy or would exceed its limit.
    def give(memory : T, bytesize : Int32) : Bool
      return false if @busy || bytesize == 0 || @held + bytesize > limit
      @busy = true
      begin
        (@free[bytesize] ||= [] of T) << memory
        @held += bytesize
      ensure
        @busy = false
      end
      true
    end
  end
end
