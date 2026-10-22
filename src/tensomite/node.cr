module Tensomite
  # An operation a graph node can perform.
  enum Op
    # Elementwise sum of two sources of equal shape.
    Add
    # Elementwise product of two sources of equal shape.
    Mul
    # Sum of one source over `Node#axis`, or over all elements when the axis is `nil`.
    Sum
    # Row-major copy of one source.
    Copy
  end

  # A vertex of the compute graph, producing one contiguous array of `shape`.
  #
  # A leaf node has no op and is created realised. Any other node holds an op and its
  # sources until the scheduler realises it, after which it holds only its buffer.
  class Node
    getter op : Op?
    getter srcs : Array(Tensor)
    getter shape : Array(Int32)
    getter dtype : DType
    getter device : Device
    getter axis : Int32?
    getter buffer : Buffer?

    def initialize(@op : Op, @srcs : Array(Tensor), @shape, @dtype, @device, @axis = nil)
    end

    def initialize(buffer : Buffer, @shape, @dtype, @device)
      @op = nil
      @srcs = [] of Tensor
      @buffer = buffer
    end

    # Returns the number of elements produced.
    def size : Int32
      shape.product(1)
    end

    # Returns the number of bytes produced.
    def bytesize : Int32
      size * dtype.bytesize
    end

    # Returns the buffer of a realised node; raises `Error` otherwise.
    def buffer! : Buffer
      @buffer || raise Error.new("node is not realised")
    end

    def realized? : Bool
      !@buffer.nil?
    end

    # Records the computed *buffer* and drops the sources so they can be collected.
    def realized(buffer : Buffer) : Nil
      @buffer = buffer
      @srcs = [] of Tensor
    end
  end
end
