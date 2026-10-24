module Tensomite
  # Freed device memory kept for reuse, keyed by size in bytes.
  #
  # Devices give memory back when the collector finalizes a buffer, and take it
  # before allocating anew, so a loop that builds the same graph every step, such
  # as training or decoding, stops allocating once it is warm.
  #
  # Finalizers can run on any thread, and in the middle of an allocation,
  # including one made by the pool itself. So the pool never waits: while one
  # caller is changing it, `#take` returns `nil` and `#give` returns false, and
  # the caller allocates, or frees the memory, instead.
  class BufferPool(T)
    # The most bytes the pool holds; memory beyond it is left to be freed.
    getter limit : Int64

    # Bytes held now.
    getter held = 0_i64

    @free = {} of Int32 => Array(T)
    @lock = Atomic(Bool).new(false)
    @hits = Atomic(Int64).new(0)
    @misses = Atomic(Int64).new(0)

    def initialize(@limit : Int64)
    end

    # Allocations served from the pool.
    def hits : Int64
      @hits.get
    end

    # Allocations the pool could not serve.
    def misses : Int64
      @misses.get
    end

    # Returns memory of exactly *bytesize* bytes, or `nil` when none is free.
    def take(bytesize : Int32) : T?
      memory = nil
      if lock
        begin
          if memory = @free[bytesize]?.try(&.pop?)
            @held -= bytesize
          end
        ensure
          unlock
        end
      end
      (memory ? @hits : @misses).add(1)
      memory
    end

    # Keeps *memory* of *bytesize* bytes and returns true, or returns false when
    # the pool is busy or would exceed its limit.
    def give(memory : T, bytesize : Int32) : Bool
      return false if bytesize == 0 || !lock
      begin
        return false if @held + bytesize > limit
        (@free[bytesize] ||= [] of T) << memory
        @held += bytesize
      ensure
        unlock
      end
      true
    end

    private def lock : Bool
      @lock.compare_and_set(false, true)[1]
    end

    private def unlock : Nil
      @lock.set(false)
    end
  end
end
