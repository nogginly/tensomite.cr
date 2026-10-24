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

    # Gives the `MTLBuffer` back to its device, which pools or releases it.
    def finalize
      if metal = device.as?(Metal)
        metal.retire(@mtl, bytesize)
      else
        ObjC.release(@mtl)
      end
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
  # off. Arithmetic, comparisons and reductions give the same bits as the host
  # kernels. `exp`, `log`, `tanh` and `sqrt` may differ in the last bits, and so
  # may `MatMul`, which splits `k` across threads (`#matmul_mode`); all of them
  # repeat exactly from run to run and are shape-invariant (Decisions K and S).
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
    private OTHER          = %w[copy_into_f32 sum_f32 max_f32 sum_short_f32 max_short_f32 gather_f32 scatter_add_f32 unview_f32 matmul_f32 matmul_split_f32]
    private OPTIONAL       = %w[fill_zero_f32 matmul_atomic_f32]
    private COLS           =  32
    private SLICES         =   8
    private REDUCE_THREADS = 256
    private REDUCE_CHUNK   =  64

    # How `MatMul` divides its work.
    enum MatMulMode
      # One thread per output, adding over `k` in index order: bit-identical to the
      # host, but a matrix-vector product leaves most of the GPU idle.
      Serial
      # Several threads per output, each summing fixed chunks of `k`, combined in
      # chunk order: deterministic and shape-invariant. The default.
      Split
      # As `Split`, but each thread adds its chunks into the output atomically, in
      # whatever order threads finish. Not deterministic; for measurement only.
      Atomic
    end

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

    # How `MatMul` divides its work; see `MatMulMode`.
    property matmul_mode = MatMulMode::Split

    # Where a `Metal` device's time goes, collected while `Metal#profile` is set.
    #
    # ```
    # metal.profile = Metal::Profile.new
    # model.forward(ids, cache)
    # metal.profile.try(&.encode) # => seconds spent encoding dispatches
    # ```
    class Profile
      # True when every op is committed and timed on the GPU on its own. That gives
      # GPU time per kernel, but adds a round trip per op to the wall time.
      getter? per_kernel : Bool

      # Count and GPU seconds per kernel; filled only when `#per_kernel?`.
      getter kernels = Hash(String, {Int32, Float64}).new({0, 0.0})

      # Count and wall seconds per op run on the host.
      getter host_ops = Hash(String, {Int32, Float64}).new({0, 0.0})

      # Seconds spent encoding dispatches, allocating buffers, and waiting for
      # command buffers to complete.
      property encode = 0.0
      property alloc = 0.0
      property wait = 0.0

      # Dispatches encoded and command buffers committed.
      property dispatches = 0
      property command_buffers = 0

      def initialize(@per_kernel = false)
      end

      protected def add_kernel(name : String, seconds : Float64) : Nil
        count, total = kernels[name]
        kernels[name] = {count + 1, total + seconds}
      end

      protected def add_host_op(name : String, seconds : Float64) : Nil
        count, total = host_ops[name]
        host_ops[name] = {count + 1, total + seconds}
      end
    end

    # The profile being collected, or `nil` (the default) for none.
    property profile : Profile? = nil

    @device : Void*
    @queue : Void*
    @pipelines = {} of String => Void*
    @command_buffer = Pointer(Void).null
    @encoder = Pointer(Void).null
    # Command buffers committed but not yet waited for, oldest first.
    @in_flight = [] of Void*
    # Dispatches encoded into the open command buffer.
    @encoded = 0
    # Buffers given back while GPU work was pending; they join the pool once it completes.
    @retired = [] of {Void*, Int32}
    @retiring = false

    # Freed `MTLBuffer`s kept for reuse; see `BufferPool`. Holds up to 1 GiB by
    # default. (`#pool`, inherited, holds host memory, which `Metal` does not use.)
    getter buffer_pool = BufferPool(Void*).new(1_i64 << 30)

    # How many dispatches go into a command buffer before it is committed, so the
    # GPU starts on them while the host is still encoding the rest.
    property commit_every = 64
    # Where kernels report an index out of range: a count, the first bad id, and
    # the number of rows it was checked against.
    private getter(errors : MetalBuffer) { new_buffer(12).tap(&.bytes.fill(0)) }

    def initialize
      @device = LibMetal.MTLCreateSystemDefaultDevice
      raise DeviceError.new("no Metal device") if @device.null?
      device = @device
      @name = ObjC.autoreleasepool { String.new(ObjC.send(Pointer(UInt8), ObjC.send(Pointer(Void), device, "name"), "UTF8String")) }
      @queue = ObjC.send(Pointer(Void), @device, "newCommandQueue")
      compile
    end

    def alloc(bytesize : Int32) : Buffer
      return new_buffer(bytesize) unless profile = @profile
      start = Time.instant
      new_buffer(bytesize).tap { profile.alloc += (Time.instant - start).total_seconds }
    end

    # Returns a buffer of *bytesize* bytes, from the pool when one is free.
    private def new_buffer(bytesize : Int32) : MetalBuffer
      mtl = buffer_pool.take(bytesize) ||
            ObjC.send(Pointer(Void), @device, "newBufferWithLength:options:", Math.max(bytesize, 4).to_u64, STORAGE_SHARED)
      raise DeviceError.new("cannot allocate #{bytesize} bytes") if mtl.null?
      MetalBuffer.new(self, mtl, bytesize)
    end

    # Takes back the `MTLBuffer` of a collected buffer. It joins the pool at once
    # when no GPU work is pending, otherwise when that work completes, because the
    # GPU may still be reading it; it is released when the pool cannot take it.
    protected def retire(mtl : Void*, bytesize : Int32) : Nil
      if idle?
        ObjC.release(mtl) unless buffer_pool.give(mtl, bytesize)
      elsif @retiring
        ObjC.release(mtl)
      else
        @retiring = true
        begin
          @retired << {mtl, bytesize}
        ensure
          @retiring = false
        end
      end
    end

    private def idle? : Bool
      @command_buffer.null? && @in_flight.empty?
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
      profile = @profile
      start = Time.instant if profile
      if encode(node, metal(output))
        @gpu_ops += 1
        if profile && start
          profile.encode += (Time.instant - start).total_seconds
          flush(node.op.to_s) if profile.per_kernel?
        end
      else
        flush
        start = Time.instant if profile
        super
        @host_ops += 1
        profile.add_host_op("#{node.op} #{node.dtype}", (Time.instant - start).total_seconds) if profile && start
      end
    end

    # Encodes *node* as GPU work and returns true, or returns false when it has no
    # Metal kernel: an op not yet ported, a dtype other than `F32`, a view of more
    # than 8 dimensions, or a view `Unview` cannot invert.
    private def encode(node : Node, output : MetalBuffer) : Bool
      return false unless node.dtype.f32? && node.srcs.all? { |src| src.ndim <= MAX_DIMS }
      srcs = node.srcs
      case node.op
      when Op::Sum, Op::Max
        encode_reduce(node, output)
      when Op::Cat
        encode_cat(node, output)
      when Op::Gather, Op::ScatterAdd
        encode_rows(node, output)
      when Op::Unview
        return false unless encode_unview(node, output)
      when Op::MatMul
        encode_matmul(node, output)
      else
        name = srcs.size == 2 ? BINARY[node.op]? : UNARY[node.op]?
        return false unless name
        dispatch(name, node.size, srcs.map { |src| mtl(src) } << output.mtl,
          srcs.map { |src| view_args(src.view) } << u32(node.size))
      end
      true
    end

    # Sum and Max, folding each output's run of the source permuted so the reduced
    # axis is last: one threadgroup per output, with a thread per chunk of the run
    # up to `REDUCE_THREADS`, or one thread per output when the run is one chunk.
    private def encode_reduce(node : Node, output : MetalBuffer) : Nil
      src = node.srcs.first
      view = src.view
      run = view.size
      if axis = node.axis
        view = view.permute((0...view.ndim).reject(axis) + [axis])
        run = view.shape.last
      end
      name = node.op == Op::Sum ? "sum" : "max"
      chunks = (run + REDUCE_CHUNK - 1) // REDUCE_CHUNK
      if chunks <= 1
        dispatch("#{name}_short_f32", node.size, [mtl(src), output.mtl], [view_args(view), u32(run), u32(node.size)])
      else
        threads = Math.min(chunks, REDUCE_THREADS)
        dispatch_groups("#{name}_f32", {node.size, 1}, {threads, 1}, [mtl(src), output.mtl], [view_args(view), u32(run)])
      end
    end

    # MatMul, divided as `#matmul_mode` says.
    #
    # `Split` and `Atomic` launch one threadgroup per output row and block of
    # `COLS` columns, with `SLICES` threads per column, each summing chunks of `k`.
    private def encode_matmul(node : Node, output : MetalBuffer) : Nil
      a, b = node.srcs
      m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
      dims = Slice[m.to_u32, k.to_u32, n.to_u32].to_unsafe_bytes
      buffers = [mtl(a), mtl(b), output.mtl]
      constants = [view_args(a.view), view_args(b.view), dims]
      case matmul_mode
      in .serial?
        dispatch("matmul_f32", node.size, buffers, constants << u32(node.size))
      in .split?
        dispatch_groups("matmul_split_f32", {(n + COLS - 1) // COLS, node.size // n}, {COLS, SLICES}, buffers, constants)
      in .atomic?
        raise DeviceError.new("atomic matmul needs Metal 3") unless @pipelines.has_key?("matmul_atomic_f32")
        dispatch("fill_zero_f32", node.size, [output.mtl], [u32(node.size)])
        dispatch_groups("matmul_atomic_f32", {(n + COLS - 1) // COLS, node.size // n}, {COLS, SLICES}, buffers, constants)
      end
    end

    # Cat: one dispatch per source, copying it into its slice of the output.
    private def encode_cat(node : Node, output : MetalBuffer) : Nil
      axis = node.axis || raise Error.new("cat needs an axis")
      out_view = View.contiguous(node.shape)
      start = 0
      node.srcs.each do |src|
        stop = start + src.shape[axis]
        region = out_view.slice(axis, start, stop)
        dispatch("copy_into_f32", src.size, [mtl(src), output.mtl], [view_args(src.view), view_args(region), u32(src.size)])
        start = stop
      end
    end

    # Gather, one thread per output element, and ScatterAdd, one thread per column.
    # Both report an id out of range through the error buffer (see `#flush`).
    private def encode_rows(node : Node, output : MetalBuffer) : Nil
      rows_src, ids = node.srcs
      rows, width = node.op == Op::Gather ? rows_src.shape : node.shape
      buffers = [mtl(rows_src), mtl(ids), output.mtl, errors.mtl]
      constants = [view_args(rows_src.view), view_args(ids.view), u32(width), i32(rows)]
      if node.op == Op::Gather
        dispatch("gather_f32", node.size, buffers, constants << u32(node.size))
      else
        dispatch("scatter_add_f32", width, buffers, constants << u32(ids.size))
      end
    end

    # Unview: one thread per output position, summing what the view reads from it.
    # Returns false for a view whose positions are not each read by a regular grid
    # of indices; see `#inverse_args`.
    private def encode_unview(node : Node, output : MetalBuffer) : Bool
      view = node.source_view || raise Error.new("unview needs a source view")
      return false unless view.ndim <= MAX_DIMS && (inverse = inverse_args(view))
      src = node.srcs.first
      dispatch("unview_f32", node.size, [mtl(src), output.mtl], [view_args(src.view), view_args(view), inverse, u32(node.size)])
      true
    end

    # Returns the kernels' `Inverse` struct for *view*: the dimensions with a
    # nonzero stride and more than one index, largest stride first, then the rest.
    #
    # Returns `nil` unless each of those strides exceeds the furthest the smaller
    # ones can reach together, which makes the decomposition of a position unique.
    # Every view tensomite builds satisfies it; overlapping views run on the host.
    private def inverse_args(view : View) : Bytes?
      dims = (0...view.ndim)
      strided = dims.select { |dim| view.strides[dim] != 0 && view.shape[dim] > 1 }.sort_by! { |dim| -view.strides[dim] }
      reach = 0
      strided.reverse_each do |dim|
        stride = view.strides[dim]
        return unless stride > reach
        reach += stride * (view.shape[dim] - 1)
      end
      repeated = dims.reject { |dim| strided.includes?(dim) }
      args = Slice(Int32).new(2 + 2 * MAX_DIMS)
      args[0] = strided.size
      strided.each_with_index { |dim, k| args[1 + k] = dim }
      args[1 + MAX_DIMS] = repeated.size
      repeated.each_with_index { |dim, k| args[2 + MAX_DIMS + k] = dim }
      args.to_unsafe_bytes
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
        (BINARY.values + UNARY.values + OTHER + OPTIONAL).each do |name|
          function = ObjC.send(Pointer(Void), library, "newFunctionWithName:", ObjC.string(name))
          next if function.null? && OPTIONAL.includes?(name)
          raise DeviceError.new("no Metal kernel #{name}") if function.null?
          pipeline = ObjC.send(Pointer(Void), @device, "newComputePipelineStateWithFunction:error:", function, error)
          ObjC.release(function)
          raise DeviceError.new("cannot build #{name}: #{ObjC.describe(error.value)}") if pipeline.null?
          @pipelines[name] = pipeline
        end
        ObjC.release(library)
      end
    end

    # Encodes one dispatch of kernel *name* over *threads* threads, binding
    # *buffers* from index 0 and then *constants*, each copied into the command.
    private def dispatch(name : String, threads : Int32, buffers : Array(Void*), constants : Array(Bytes)) : Nil
      return if threads == 0
      pipeline = bind(name, buffers, constants)
      width = Math.min(ObjC.send(UInt64, pipeline, "maxTotalThreadsPerThreadgroup"), 256_u64)
      grid = StaticArray[threads.to_u64, 1_u64, 1_u64]
      group = StaticArray[width, 1_u64, 1_u64]
      ObjC.send(Nil, open_encoder, "dispatchThreads:threadsPerThreadgroup:", grid.to_unsafe, group.to_unsafe)
      dispatched
    end

    # Encodes one dispatch of kernel *name* over a 2-D grid of *groups*
    # threadgroups, each of *threads* threads, binding arguments as `#dispatch` does.
    private def dispatch_groups(name : String, groups : {Int32, Int32}, threads : {Int32, Int32},
                                buffers : Array(Void*), constants : Array(Bytes)) : Nil
      return if groups[0] == 0 || groups[1] == 0
      bind(name, buffers, constants)
      grid = StaticArray[groups[0].to_u64, groups[1].to_u64, 1_u64]
      group = StaticArray[threads[0].to_u64, threads[1].to_u64, 1_u64]
      ObjC.send(Nil, open_encoder, "dispatchThreadgroups:threadsPerThreadgroup:", grid.to_unsafe, group.to_unsafe)
      dispatched
    end

    # Commits the open command buffer once it holds `#commit_every` dispatches.
    private def dispatched : Nil
      @encoded += 1
      commit if @encoded >= commit_every
    end

    # Ends and commits the open command buffer, if any, without waiting for it.
    private def commit : Nil
      return if @command_buffer.null?
      ObjC.send(Nil, @encoder, "endEncoding")
      ObjC.release(@encoder)
      ObjC.send(Nil, @command_buffer, "commit")
      @in_flight << @command_buffer
      @encoder = @command_buffer = Pointer(Void).null
      @encoded = 0
    end

    # Sets kernel *name*'s pipeline on the open encoder and binds its arguments;
    # returns the pipeline.
    private def bind(name : String, buffers : Array(Void*), constants : Array(Bytes)) : Void*
      @profile.try { |profile| profile.dispatches += 1 }
      pipeline = @pipelines[name]
      encoder = open_encoder
      ObjC.send(Nil, encoder, "setComputePipelineState:", pipeline)
      buffers.each_with_index do |buffer, index|
        ObjC.send(Nil, encoder, "setBuffer:offset:atIndex:", buffer, 0_u64, index.to_u64)
      end
      constants.each_with_index(buffers.size) do |bytes, index|
        ObjC.send(Nil, encoder, "setBytes:length:atIndex:", bytes.to_unsafe.as(Void*), bytes.size.to_u64, index.to_u64)
      end
      pipeline
    end

    # Returns *view* laid out as the kernels' `View` struct.
    private def view_args(view : View) : Bytes
      args = Slice(Int32).new(2 + 2 * MAX_DIMS)
      args[0], args[1] = view.ndim, view.offset
      view.ndim.times do |dim|
        args[2 + dim] = view.shape[dim]
        args[2 + MAX_DIMS + dim] = view.strides[dim]
      end
      args.to_unsafe_bytes
    end

    private def u32(value : Int32) : Bytes
      Slice[value.to_u32].to_unsafe_bytes
    end

    private def i32(value : Int32) : Bytes
      Slice[value].to_unsafe_bytes
    end

    private def mtl(tensor : Tensor) : Void*
      metal(tensor.node.buffer!).mtl
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

    # Commits the open command buffer, if any, and waits for every command buffer
    # in flight, in order. Then it moves retired buffers into the pool, and raises
    # `ExecutionError` if the GPU reported a failure or a kernel reported an index
    # out of range.
    #
    # With a per-kernel profile, *label* names the op whose work the command
    # buffers hold, and their GPU time is recorded under it.
    private def flush(label : String? = nil) : Nil
      commit
      return if @in_flight.empty?
      in_flight, @in_flight = @in_flight, [] of Void*
      profile = @profile
      start = Time.instant
      in_flight.each { |command_buffer| ObjC.send(Nil, command_buffer, "waitUntilCompleted") }
      if profile
        profile.wait += (Time.instant - start).total_seconds
        profile.command_buffers += in_flight.size
        if label
          gpu = in_flight.sum { |done| ObjC.send(Float64, done, "GPUEndTime") - ObjC.send(Float64, done, "GPUStartTime") }
          profile.add_kernel(label, gpu)
        end
      end
      message = ObjC.autoreleasepool do
        in_flight.compact_map do |command_buffer|
          status = ObjC.send(UInt64, command_buffer, "status")
          ObjC.describe(ObjC.send(Pointer(Void), command_buffer, "error")) unless status == STATUS_COMPLETED
        end.first?
      end
      in_flight.each { |command_buffer| ObjC.release(command_buffer) }
      recycle_retired
      raise ExecutionError.new("Metal command buffer failed: #{message}") if message
      if (reports = @errors.try(&.bytes.unsafe_slice_of(Int32))) && reports[0] != 0
        id, rows = reports[1], reports[2]
        reports.fill(0)
        raise ExecutionError.new("index #{id} out of range for #{rows} rows")
      end
    end

    # Moves the buffers retired while GPU work was pending into the pool.
    private def recycle_retired : Nil
      retired, @retired = @retired, [] of {Void*, Int32}
      retired.each { |(mtl, bytesize)| ObjC.release(mtl) unless buffer_pool.give(mtl, bytesize) }
    end

    private def metal(buffer : Buffer) : MetalBuffer
      own(buffer).as?(MetalBuffer) || raise DeviceError.new("buffer is not a Metal buffer")
    end
  end
end
