module Tensomite
  # The pure-Crystal reference device. Kernels are naive and run in a fixed order,
  # so every other device is tested against it.
  class CPU < Device
    # A buffer in host memory.
    class Buffer < Tensomite::Buffer
      getter bytes : Bytes

      def initialize(bytesize : Int32)
        super
        @bytes = Bytes.new(bytesize)
      end
    end

    # Returns the shared CPU device.
    class_getter default : CPU { CPU.new }

    def alloc(bytesize : Int32) : Tensomite::Buffer
      Buffer.new(bytesize)
    end

    def upload(buffer : Tensomite::Buffer, data : Bytes) : Nil
      data.copy_to(own(buffer).bytes)
    end

    def download(buffer : Tensomite::Buffer, data : Bytes) : Nil
      own(buffer).bytes.copy_to(data)
    end

    def map(buffer : Tensomite::Buffer) : Bytes?
      own(buffer).bytes
    end

    def execute(node : Node, output : Tensomite::Buffer) : Nil
      case node.dtype
      in .f32? then execute(Float32, node, output)
      in .i32? then execute(Int32, node, output)
      end
    end

    def sync : Nil
    end

    private def execute(type : T.class, node : Node, output : Tensomite::Buffer) : Nil forall T
      dst = elements(T, output)
      case node.op
      in Op::Add  then binary(T, node, dst) { |x, y| x + y }
      in Op::Mul  then binary(T, node, dst) { |x, y| x * y }
      in Op::Sum  then sum(T, node, dst)
      in Op::Copy then copy(T, node, dst)
      in Nil      then raise Error.new("leaf nodes are created realised")
      end
    end

    # Applies the block to each pair of elements; sources are already expanded to the output shape.
    private def binary(type : T.class, node : Node, dst : Slice(T), & : T, T -> T) : Nil forall T
      a, b = node.srcs
      xs, ys = elements(T, a), elements(T, b)
      dst.size.times do |i|
        dst[i] = yield xs[a.view.offset_of(i)], ys[b.view.offset_of(i)]
      end
    end

    # Sums along the axis by permuting it to the end, so each output element owns
    # one contiguous run of `n` logical indices, which is added left to right.
    private def sum(type : T.class, node : Node, dst : Slice(T)) : Nil forall T
      src = node.srcs.first
      xs = elements(T, src)
      view = src.view
      n = view.size
      if axis = node.axis
        rest = (0...view.ndim).reject(axis)
        view = view.permute(rest + [axis])
        n = view.shape.last
      end
      dst.size.times do |row|
        acc = T.zero
        n.times { |j| acc += xs[view.offset_of(row * n + j)] }
        dst[row] = acc
      end
    end

    private def copy(type : T.class, node : Node, dst : Slice(T)) : Nil forall T
      src = node.srcs.first
      xs = elements(T, src)
      dst.size.times { |i| dst[i] = xs[src.view.offset_of(i)] }
    end

    private def elements(type : T.class, tensor : Tensor) : Slice(T) forall T
      elements(T, tensor.node.buffer!)
    end

    private def elements(type : T.class, buffer : Tensomite::Buffer) : Slice(T) forall T
      own(buffer).bytes.unsafe_slice_of(T)
    end

    private def own(buffer : Tensomite::Buffer) : Buffer
      buffer.as?(Buffer) || raise DeviceError.new("buffer does not belong to the CPU device")
    end
  end
end
