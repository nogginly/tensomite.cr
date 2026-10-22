module Tensomite
  # An operation a graph node can perform. Devices implement every member.
  enum Op
    # Elementwise `a + b`; sources have the output shape.
    Add
    # Elementwise `a - b`.
    Sub
    # Elementwise `a * b`.
    Mul
    # Elementwise `a / b`, float types only.
    Div
    # Elementwise larger of `a` and `b`.
    Maximum
    # Elementwise `e^x`, float types only.
    Exp
    # Elementwise natural logarithm, float types only.
    Log
    # Elementwise hyperbolic tangent, float types only.
    Tanh
    # Elementwise square root, float types only.
    Sqrt
    # Sum over `Node#axis`, or over all elements when the axis is `nil`.
    Sum
    # Largest element over `Node#axis`, or over all elements when the axis is `nil`.
    Max
    # Row-major copy of one source.
    Copy
    # All sources joined along `Node#axis`, in order.
    Cat
    # Rows of source 0, a 2-D table, selected by the `I32` indices in source 1.
    Gather
    # Matrix product of `[..., M, K]` and `[..., K, N]`; batch dimensions are already broadcast.
    MatMul
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
      dtype.bytes_for(size)
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
