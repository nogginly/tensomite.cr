module Tensomite
  # A count of bytes or a position in a buffer: 64-bit, so that a buffer can
  # exceed 2 GiB.
  alias Index = Int64

  # A run of host memory, by pointer and size in bytes. Unlike `Bytes`, whose
  # size is an `Int32`, it may exceed 2 GiB. It does not own the memory.
  #
  # ```
  # memory = Memory.allocate(3_i64 << 30)       # 3 GiB, zeroed
  # memory.unsafe_slice_of(Float32)[0] = 1.5f32 # a typed view of its elements
  # memory[0, 4].bytes                          # => Bytes[0, 0, 192, 63]
  # ```
  struct Memory
    getter pointer : Pointer(UInt8)
    getter size : Index

    def initialize(@pointer : Pointer(UInt8), size : Int)
      raise ArgumentError.new("negative size #{size}") if size.negative?
      @size = size.to_i64
    end

    # Returns the memory *bytes* spans.
    def self.new(bytes : Bytes) : Memory
      new(bytes.to_unsafe, bytes.size)
    end

    # Returns *size* bytes of new, zeroed memory, which the garbage collector
    # frees once nothing points into it.
    def self.allocate(size : Int) : Memory
      new(Pointer(UInt8).malloc(size.to_u64), size)
    end

    # Returns an empty run.
    def self.empty : Memory
      new(Pointer(UInt8).null, 0)
    end

    def empty? : Bool
      size == 0
    end

    def to_unsafe : Pointer(UInt8)
      pointer
    end

    # Returns *count* bytes from *start*; raises `IndexError` outside this run.
    def [](start : Int, count : Int) : Memory
      unless 0 <= start && 0 <= count && start.to_i64 + count <= size
        raise IndexError.new("#{count} bytes from #{start} leave #{size} bytes")
      end
      Memory.new(pointer + start, count)
    end

    # Returns the run as `Bytes`; raises `ArgumentError` when it is 2 GiB or more.
    def bytes : Bytes
      raise ArgumentError.new("#{size} bytes do not fit in Bytes") if size > Int32::MAX
      Bytes.new(pointer, size.to_i32)
    end

    # Returns the run as elements of *type*: as many whole ones as fit, at most
    # `Int32::MAX`, past which it raises `ArgumentError`.
    def unsafe_slice_of(type : T.class) : Slice(T) forall T
      count = size // sizeof(T)
      raise ArgumentError.new("#{count} #{T} elements do not fit in a Slice") if count > Int32::MAX
      Slice.new(pointer.as(T*), count.to_i32)
    end

    # Copies this run into the start of *target*, which must be at least as long.
    def copy_to(target : Memory) : Nil
      raise IndexError.new("#{size} bytes do not fit in #{target.size}") if target.size < size
      pointer.copy_to(target.pointer, size)
    end

    # Writes the run to *io*, in pieces small enough for `Bytes`.
    def write_to(io : IO) : Nil
      piece = 1_i64 << 30
      start = 0_i64
      while start < size
        count = Math.min(piece, size - start)
        io.write(Bytes.new(pointer + start, count.to_i32))
        start += count
      end
    end

    # Sets every byte to *value*.
    def fill(value : UInt8) : self
      Intrinsics.memset(pointer.as(Void*), value, size.to_u64, false) unless empty?
      self
    end
  end
end
