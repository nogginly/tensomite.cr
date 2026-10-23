require "../tensomite"

{% unless flag?(:darwin) && flag?(:aarch64) %}
  {% raise "tensomite/metal needs macOS on Apple Silicon" %}
{% end %}

require "./metal/objc"

# Opt-in Metal support: `require "tensomite/metal"` links the Metal framework and
# adds the `Tensomite::Metal` device. Apple Silicon only.
module Tensomite
  @[Link(framework: "Metal")]
  lib LibMetal
    fun MTLCreateSystemDefaultDevice : Void*
  end

  # A Metal buffer in shared storage, which the CPU and GPU both address directly.
  class MetalBuffer < HostBuffer
    # The `MTLBuffer`, owned by this buffer.
    getter mtl : Void*

    def initialize(device : Device, @mtl : Void*, bytesize : Int32)
      super(device, Bytes.new(ObjC.send(Pointer(UInt8), @mtl, "contents"), bytesize))
    end

    def finalize
      ObjC.release(@mtl)
    end
  end

  # The GPU of an Apple Silicon Mac, through Metal.
  #
  # Memory is unified, so buffers are shared: `#map` returns them directly, and
  # uploads and downloads are copies in host memory. Ops with a Metal kernel are
  # encoded into one command buffer, which is committed when the host next needs
  # the results; every other op waits for the GPU and runs the host kernel on the
  # same memory. Porting an op to Metal therefore changes where it runs, never
  # whether it runs.
  #
  # Kernels are compiled from MSL source when the device is created, with fast math
  # off. Arithmetic and comparisons give the same bits as the host kernels;
  # `exp`, `log`, `tanh` and `sqrt` may differ in the last bits (level 3 of
  # Decision K).
  class Metal < HostDevice
    private SOURCE           = {{ read_file("#{__DIR__}/metal/kernels.metal") }}
    private MAX_DIMS         =     8
    private STORAGE_SHARED   = 0_u64
    private STATUS_COMPLETED = 4_u64

    private BINARY = {
      Op::Add => "add_f32", Op::Sub => "sub_f32", Op::Mul => "mul_f32", Op::Div => "div_f32",
      Op::Maximum => "maximum_f32", Op::GreaterEqual => "greater_equal_f32",
    }
    private UNARY = {
      Op::Exp => "exp_f32", Op::Log => "log_f32", Op::Tanh => "tanh_f32", Op::Sqrt => "sqrt_f32",
      Op::Copy => "copy_f32",
    }

    # Returns true when the system has a Metal GPU. Some virtual machines, such as
    # hosted CI runners, do not.
    def self.available? : Bool
      device = LibMetal.MTLCreateSystemDefaultDevice
      ObjC.release(device)
      !device.null?
    end

    # Returns the shared Metal device; raises `DeviceError` without a GPU.
    class_getter default : Metal { Metal.new }

    # Returns the GPU's name, such as "Apple M5 Pro".
    getter name : String

    # Returns how many ops have run on the GPU and on the host since creation.
    getter gpu_ops = 0
    getter host_ops = 0

    @device : Void*
    @queue : Void*
    @pipelines = {} of String => Void*
    @command_buffer = Pointer(Void).null
    @encoder = Pointer(Void).null

    def initialize
      @device = LibMetal.MTLCreateSystemDefaultDevice
      raise DeviceError.new("no Metal device") if @device.null?
      device = @device
      @name = ObjC.autoreleasepool { String.new(ObjC.send(Pointer(UInt8), ObjC.send(Pointer(Void), device, "name"), "UTF8String")) }
      @queue = ObjC.send(Pointer(Void), @device, "newCommandQueue")
      compile
    end

    def alloc(bytesize : Int32) : Buffer
      mtl = ObjC.send(Pointer(Void), @device, "newBufferWithLength:options:", Math.max(bytesize, 4).to_u64, STORAGE_SHARED)
      raise DeviceError.new("cannot allocate #{bytesize} bytes") if mtl.null?
      MetalBuffer.new(self, mtl, bytesize)
    end

    def upload(buffer : Buffer, data : Bytes) : Nil
      flush
      super
    end

    def download(buffer : Buffer, data : Bytes) : Nil
      flush
      super
    end

    def map(buffer : Buffer) : Bytes?
      flush
      own(buffer).bytes
    end

    # Copies *bytes* into a new Metal buffer; a file mapping cannot be handed to the GPU.
    def adopt(bytes : Bytes, owner : MappedFile) : Buffer
      alloc(bytes.size).tap { |buffer| upload(buffer, bytes) }
    end

    def sync : Nil
      flush
    end

    # Metal kernels compute each element independently of the shape around it, and
    # so do the host kernels that run the ops not yet ported.
    def shape_invariant? : Bool
      true
    end

    def execute(node : Node, output : Buffer) : Nil
      if (name = kernel_name(node)) && node.srcs.all? { |src| src.ndim <= MAX_DIMS }
        encode(name, node, metal(output))
        @gpu_ops += 1
      else
        flush
        super
        @host_ops += 1
      end
    end

    # Returns the Metal kernel for *node*, or `nil` when it runs on the host.
    private def kernel_name(node : Node) : String?
      return unless node.dtype.f32?
      node.srcs.size == 2 ? BINARY[node.op]? : UNARY[node.op]?
    end

    # Compiles the kernel source and builds a compute pipeline for every kernel in it.
    private def compile : Nil
      ObjC.autoreleasepool do
        options = ObjC.send(Pointer(Void), ObjC.send(Pointer(Void), ObjC.cls("MTLCompileOptions"), "alloc"), "init")
        ObjC.send(Nil, options, "setFastMathEnabled:", 0_u32)
        error = Pointer(Void*).malloc(1)
        library = ObjC.send(Pointer(Void), @device, "newLibraryWithSource:options:error:", ObjC.string(SOURCE), options, error)
        ObjC.release(options)
        raise DeviceError.new("cannot compile Metal kernels: #{ObjC.describe(error.value)}") if library.null?
        (BINARY.values + UNARY.values).each do |name|
          function = ObjC.send(Pointer(Void), library, "newFunctionWithName:", ObjC.string(name))
          raise DeviceError.new("no Metal kernel #{name}") if function.null?
          pipeline = ObjC.send(Pointer(Void), @device, "newComputePipelineStateWithFunction:error:", function, error)
          ObjC.release(function)
          raise DeviceError.new("cannot build #{name}: #{ObjC.describe(error.value)}") if pipeline.null?
          @pipelines[name] = pipeline
        end
        ObjC.release(library)
      end
    end

    # Encodes one dispatch of kernel *name* for *node*: sources, output, each
    # source's view, then the element count, one thread per output element.
    private def encode(name : String, node : Node, output : MetalBuffer) : Nil
      count = node.size
      return if count == 0
      pipeline = @pipelines[name]
      encoder = open_encoder
      ObjC.send(Nil, encoder, "setComputePipelineState:", pipeline)
      index = 0_u64
      node.srcs.each do |src|
        ObjC.send(Nil, encoder, "setBuffer:offset:atIndex:", metal(src.node.buffer!).mtl, 0_u64, index)
        index += 1
      end
      ObjC.send(Nil, encoder, "setBuffer:offset:atIndex:", output.mtl, 0_u64, index)
      index += 1
      node.srcs.each do |src|
        view = view_args(src.view)
        ObjC.send(Nil, encoder, "setBytes:length:atIndex:", view.to_unsafe.as(Void*), view.bytesize.to_u64, index)
        index += 1
      end
      n = count.to_u32
      ObjC.send(Nil, encoder, "setBytes:length:atIndex:", pointerof(n).as(Void*), 4_u64, index)
      width = Math.min(ObjC.send(UInt64, pipeline, "maxTotalThreadsPerThreadgroup"), 256_u64)
      grid = StaticArray[count.to_u64, 1_u64, 1_u64]
      group = StaticArray[width, 1_u64, 1_u64]
      ObjC.send(Nil, encoder, "dispatchThreads:threadsPerThreadgroup:", grid.to_unsafe, group.to_unsafe)
    end

    # Returns *view* laid out as the kernels' `View` struct.
    private def view_args(view : View) : Slice(Int32)
      args = Slice(Int32).new(2 + 2 * MAX_DIMS)
      args[0], args[1] = view.ndim, view.offset
      view.ndim.times do |dim|
        args[2 + dim] = view.shape[dim]
        args[2 + MAX_DIMS + dim] = view.strides[dim]
      end
      args
    end

    # Returns the open compute encoder, starting a command buffer when there is none.
    private def open_encoder : Void*
      if @encoder.null?
        ObjC.autoreleasepool do
          @command_buffer = ObjC.retain(ObjC.send(Pointer(Void), @queue, "commandBuffer"))
          @encoder = ObjC.retain(ObjC.send(Pointer(Void), @command_buffer, "computeCommandEncoder"))
        end
      end
      @encoder
    end

    # Commits the open command buffer, if any, and waits for it; raises
    # `ExecutionError` if the GPU reports a failure.
    private def flush : Nil
      return if @command_buffer.null?
      command_buffer = @command_buffer
      ObjC.send(Nil, @encoder, "endEncoding")
      ObjC.release(@encoder)
      @encoder = @command_buffer = Pointer(Void).null
      ObjC.send(Nil, command_buffer, "commit")
      ObjC.send(Nil, command_buffer, "waitUntilCompleted")
      message = ObjC.autoreleasepool do
        status = ObjC.send(UInt64, command_buffer, "status")
        ObjC.describe(ObjC.send(Pointer(Void), command_buffer, "error")) unless status == STATUS_COMPLETED
      end
      ObjC.release(command_buffer)
      raise ExecutionError.new("Metal command buffer failed: #{message}") if message
    end

    private def metal(buffer : Buffer) : MetalBuffer
      own(buffer).as?(MetalBuffer) || raise DeviceError.new("buffer is not a Metal buffer")
    end
  end
end
