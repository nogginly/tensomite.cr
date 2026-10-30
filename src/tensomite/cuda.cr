require "../tensomite"

{% unless flag?(:cuda) %}
  {% raise "tensomite/cuda needs -Dcuda, and the CUDA driver and NVRTC to link against" %}
{% end %}

require "./cuda/lib"

# Opt-in CUDA support: `require "tensomite/cuda"`, built with `-Dcuda`, links the
# NVIDIA driver (`libcuda`) and NVRTC (`libnvrtc`). If NVRTC is outside the
# linker's search path, give its directory in `LIBRARY_PATH`:
#
#     LIBRARY_PATH=/usr/local/cuda-13.1/targets/sbsa-linux/lib crystal build -Dcuda app.cr
module Tensomite
  # A buffer in a CUDA GPU's memory, with a host copy made when a host kernel
  # needs one.
  #
  # Device memory always holds the current data. The host copy is valid until
  # the device memory is written again: by an upload, or, once ops have CUDA
  # kernels, by a kernel.
  class CUDABuffer < HostBuffer
    # The device memory.
    getter pointer : LibCUDA::DevicePtr
    @copy : Bytes? = nil
    @copy_valid = false
    # :nodoc:
    property? writing = false

    def initialize(device : CUDA, @pointer : LibCUDA::DevicePtr, bytesize : Int32)
      super(device, bytesize, Bytes.empty)
    end

    # Returns the host copy, current: downloaded when the device memory has
    # changed since, except while a host kernel writes this buffer.
    def bytes : Bytes
      copy = @copy ||= cuda.host_bytes(bytesize)
      unless @copy_valid || writing?
        cuda.context.download(pointer, copy) unless bytesize == 0
        @copy_valid = true
      end
      copy
    end

    # Returns the host copy if it is current, otherwise `nil`.
    def current_copy : Bytes?
      @copy if @copy_valid
    end

    # Notes that the device memory was written, so the host copy is stale.
    def device_written : Nil
      @copy_valid = false
    end

    # Uploads the host copy, which a host kernel has just written.
    def host_written : Nil
      copy = @copy || raise DeviceError.new("no host copy to upload")
      cuda.context.upload(pointer, copy) unless bytesize == 0
      @copy_valid = true
    end

    # Gives the host copy to the device's host pool and the device memory to the
    # device, which reuses or frees it.
    def finalize
      @copy.try { |copy| cuda.recycle(copy) }
      cuda.retire(pointer, bytesize)
    end

    private def cuda : CUDA
      device.as(CUDA)
    end
  end

  # An NVIDIA GPU through CUDA's driver API, with kernels compiled from CUDA C++
  # by NVRTC when they are first needed.
  #
  # Memory is the GPU's own: `#map` returns `nil`, and data moves through
  # `#upload` and `#download`, even on a GPU that shares memory with the CPU,
  # so one code path serves both (Decision AS).
  #
  # Ops with a CUDA kernel (`cuda/kernels.cu`, and `Tensomite.kernel` macros)
  # run on the GPU, queued in order on the default stream; copies to and from
  # the host wait for the kernels before them. Every other op runs the host
  # kernel: its sources are downloaded to host copies, and its output is
  # computed there and uploaded. Arithmetic, comparisons and reductions of up to
  # one chunk give the host's bits; `exp`, `log` and `tanh` may differ in the
  # last bits, and so may longer sums and `MatMul`, which add in fixed chunks
  # (Decisions K, S and X), as on Metal.
  #
  # Work is issued from the thread that created the device, on which its context
  # is current.
  class CUDA < HostDevice
    private PRELUDE        = {{ read_file("#{__DIR__}/cuda/prelude.cu") }}
    private SOURCE         = PRELUDE + {{ read_file("#{__DIR__}/cuda/kernels.cu") }}
    private MAX_DIMS       =   8
    private REDUCE_THREADS = 256
    private REDUCE_CHUNK   =  16
    private BLOCK          = 256
    private COLS           =  32
    private SLICES         =   8
    private TILE_PATCH     =   4
    private DECODE_COLS    = 128
    private DECODE_ROWS    =  16
    private CHUNK          =  64

    # How `MatMul` divides its work; the kernels are Metal's (see `Metal::MatMulMode`).
    enum MatMulMode
      # One thread per output, adding over `k` in index order: bit-identical to the host.
      Serial
      # Several threads per output, each summing fixed chunks of `k`, combined in
      # chunk order: deterministic and shape-invariant.
      Split
      # Blocks of outputs per thread block, with tiles of both operands in shared
      # memory (`#matmul_tile`): bit-identical to `Split`.
      Tiled
      # Two kernels: blocks of columns per chunk of `k` writing each chunk's sum,
      # then a thread per output adding them in chunk order. Bit-identical to
      # `Split`, with many more blocks when there are few rows.
      TwoPass
      # `Tiled` for products of at least `TILED_ROWS` rows. Below that, as in
      # decoding one token, `Split` when b's columns are contiguous in memory and
      # `TwoPass` when they are not, as in a transposed b. Bit-identical to all
      # three. The default.
      Auto
    end

    # The outputs a block of the tiled matmul computes, rows by columns.
    enum MatMulTile
      T32x32
      T64x64
      T32x64

      # Returns the block's rows and columns.
      def block : {Int32, Int32}
        case self
        in .t32x32? then {32, 32}
        in .t64x64? then {64, 64}
        in .t32x64? then {32, 64}
        end
      end

      protected def kernel : String
        rows, cols = block
        "matmul_tiled_#{rows}x#{cols}_f32"
      end
    end

    # The fewest rows for which `MatMulMode::Auto` uses the tiled matmul.
    TILED_ROWS = 16

    private BINARY = {
      Op::Add => "add_f32", Op::Sub => "sub_f32", Op::Mul => "mul_f32", Op::Div => "div_f32",
      Op::Maximum => "maximum_f32", Op::GreaterEqual => "greater_equal_f32",
    }
    private UNARY = {
      Op::Exp => "exp_f32", Op::Log => "log_f32", Op::Tanh => "tanh_f32", Op::Sqrt => "sqrt_f32",
      Op::Copy => "copy_f32",
    }

    # A view as the kernels' `View` struct: ndim, offset, shape and strides of
    # `MAX_DIMS` each, and whether it is dense.
    alias ViewArgs = StaticArray(Int32, 19)

    # The kernels' `Inverse` struct; see `#inverse_args`.
    alias InverseArgs = StaticArray(Int32, 18)

    # Raised when NVRTC rejects a kernel's source; the message holds its log.
    class CompileError < DeviceError; end

    # A GPU, as the driver describes it.
    record GPU, ordinal : Int32, name : String, memory : UInt64,
      compute_capability : {Int32, Int32}, multiprocessors : Int32, integrated : Bool do
      # The compute capability as one number, such as 75 for 7.5.
      def arch : Int32
        compute_capability[0] * 10 + compute_capability[1]
      end
    end

    @@initialised : Bool? = nil

    # Returns the device on the first GPU; raises `DeviceError` if there is none.
    class_getter default : CUDA { CUDA.new(CUDA.gpus.first? || raise DeviceError.new("no CUDA GPU")) }

    getter gpu : GPU
    getter context : Context

    # Freed device memory kept for reuse; see `BufferPool`. Holds up to 1 GiB by
    # default. (`#pool`, inherited, holds host copies.)
    getter buffer_pool = BufferPool(LibCUDA::DevicePtr).new(1_i64 << 30)

    # Returns the number of uploads since creation.
    getter uploads = 0

    # Returns the number of downloads since creation.
    getter downloads = 0

    # Returns how many ops have run on the GPU and on the host since creation.
    getter gpu_ops = 0
    getter host_ops = 0

    # How `MatMul` divides its work; see `MatMulMode`.
    property matmul_mode = MatMulMode::Auto

    # The block size of the tiled matmul; see `MatMulTile`.
    property matmul_tile = MatMulTile::T64x64

    # Kernels defined with `Tensomite.kernel`, each compiled on first use, by name.
    @macro_kernels = {} of String => Kernels

    # Where a `CUDA` device's time goes, collected while `CUDA#profile` is set.
    #
    # ```
    # cuda.profile = CUDA::Profile.new
    # model.forward(ids, cache)
    # cuda.profile.try(&.launch) # => seconds spent launching kernels
    # ```
    class Profile
      # True when every op is timed on the GPU on its own, between two events,
      # waiting for it to finish. That gives GPU time per op, but serialises the
      # host and the GPU, so the wall time grows.
      getter? per_kernel : Bool

      # Count and GPU seconds per op; filled only when `#per_kernel?`.
      getter kernels = Hash(String, {Int32, Float64}).new({0, 0.0})

      # Count and wall seconds per op run on the host.
      getter host_ops = Hash(String, {Int32, Float64}).new({0, 0.0})

      # Seconds spent preparing and queuing kernels, allocating buffers, copying
      # to and from the host, and waiting for the GPU at `sync`.
      property launch = 0.0
      property alloc = 0.0
      property transfer = 0.0
      property wait = 0.0

      # Kernel launches, uploads and downloads.
      property launches = 0
      property uploads = 0
      property downloads = 0

      # True when `#kernels` keys each op by its shapes as well as its name: a
      # matmul's two sources, `MatMul [64,768]·[768,2304]`, and any other op's
      # output. Takes effect only with `#per_kernel?`.
      getter? shapes : Bool

      def initialize(@per_kernel = false, @shapes = false)
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

    @events : {Void*, Void*}? = nil

    @retired = [] of {LibCUDA::DevicePtr, Int32}
    @scratch : CUDABuffer? = nil
    @retired_lock = Atomic(Bool).new(false)
    @kernels : Kernels? = nil
    @kernels_loaded = false
    @errors : LibCUDA::DevicePtr? = nil
    @errors_pending = false

    def initialize(@gpu : GPU)
      @context = Context.new(gpu)
    end

    # Returns the hand-written kernels, compiled and loaded on first use, or
    # `nil` when the GPU cannot load them, as the host stub cannot.
    def kernels : Kernels?
      unless @kernels_loaded
        @kernels_loaded = true
        @kernels = context.load?(CUDA.compile(SOURCE, CUDA.target_arch(gpu), "kernels.cu"))
      end
      @kernels
    end

    # The GPU's name, such as `NVIDIA GB10`.
    def name : String
      gpu.name
    end

    # Returns a buffer of *bytesize* bytes of device memory, from the pool when
    # one is free.
    def alloc(bytesize : Int32) : Buffer
      start = Time.instant
      recycle_retired
      pointer = buffer_pool.take(bytesize) || context.alloc(Math.max(bytesize, 4))
      @profile.try { |profile| profile.alloc += (Time.instant - start).total_seconds }
      CUDABuffer.new(self, pointer, bytesize)
    end

    def map(buffer : Buffer) : Bytes?
      cuda(buffer)
      nil
    end

    def upload(buffer : Buffer, data : Bytes) : Nil
      target = cuda(buffer)
      @uploads += 1
      start = Time.instant
      context.upload(target.pointer, data) unless data.empty?
      @profile.try do |profile|
        profile.uploads += 1
        profile.transfer += (Time.instant - start).total_seconds
      end
      target.device_written
    end

    def download(buffer : Buffer, data : Bytes) : Nil
      source = cuda(buffer)
      @downloads += 1
      start = Time.instant
      if copy = source.current_copy
        copy[0, data.size].copy_to(data)
      elsif !data.empty?
        context.download(source.pointer, data)
      end
      @profile.try do |profile|
        profile.downloads += 1
        profile.transfer += (Time.instant - start).total_seconds
      end
    end

    # Copies *bytes* to a new buffer in device memory; a file mapping cannot be
    # shared with the GPU.
    def adopt(bytes : Bytes, owner : MappedFile) : Buffer
      alloc(bytes.size).tap { |buffer| upload(buffer, bytes) }
    end

    # Queues the CUDA kernel for *node*, or, for an op without one, runs the
    # host kernel on host copies of its sources and uploads its output.
    def execute(node : Node, output : Buffer) : Nil
      target = cuda(output)
      if (kernels = self.kernels) && launch_profiled(kernels, node, target)
        target.device_written
        @gpu_ops += 1
        return
      end
      start = Time.instant
      # A write keeps the rest of its target, so the host copy must be current
      # before the host kernel writes into it.
      target.bytes if node.op == Op::Write
      target.writing = true
      begin
        super
      ensure
        target.writing = false
      end
      target.host_written
      @host_ops += 1
      @profile.try(&.add_host_op("#{node.op} #{node.dtype}", (Time.instant - start).total_seconds))
    end

    # Waits for the GPU, reuses or frees memory given back meanwhile, and raises
    # `ExecutionError` if a kernel found an index out of range.
    def sync : Nil
      start = Time.instant
      begin
        context.synchronize
        recycle_retired
        check_errors
      ensure
        @profile.try { |profile| profile.wait += (Time.instant - start).total_seconds }
      end
    end

    # :nodoc:
    def host_bytes(bytesize : Int32) : Bytes
      pool.take(bytesize) || Bytes.new(bytesize)
    end

    # Takes back the device memory of a collected buffer. Finalizers run on any
    # thread and must never wait, so when the list is busy the memory is freed
    # at once instead, which needs this context current on that thread.
    protected def retire(pointer : LibCUDA::DevicePtr, bytesize : Int32) : Nil
      unless @retired_lock.compare_and_set(false, true)[1]
        context.free_from_any_thread(pointer)
        return
      end
      begin
        @retired << {pointer, bytesize}
      ensure
        @retired_lock.set(false)
      end
    end

    # Moves retired device memory into the pool, freeing what it cannot hold.
    private def recycle_retired : Nil
      return if @retired.empty?
      fresh = [] of {LibCUDA::DevicePtr, Int32}
      until @retired_lock.compare_and_set(false, true)[1]
        Thread.yield
      end
      retired, @retired = @retired, fresh
      @retired_lock.set(false)
      retired.each { |(pointer, bytesize)| context.free(pointer) unless buffer_pool.give(pointer, bytesize) }
    end

    private def cuda(buffer : Buffer) : CUDABuffer
      own(buffer).as?(CUDABuffer) || raise DeviceError.new("buffer is not a CUDA buffer")
    end

    # Launches *node* as `#launch` does, recording the time it takes with a
    # profile, and, with a per-kernel profile, timing it on the GPU between two
    # events and waiting for it.
    private def launch_profiled(kernels : Kernels, node : Node, output : CUDABuffer) : Bool
      return launch(kernels, node, output) unless profile = @profile
      launches = context.launches
      events = @events ||= {context.event, context.event}
      context.record(events[0]) if profile.per_kernel?
      start = Time.instant
      launched = launch(kernels, node, output)
      profile.launch += (Time.instant - start).total_seconds
      profile.launches += context.launches - launches
      if launched && profile.per_kernel?
        context.record(events[1])
        profile.add_kernel(profile_label(node, profile), context.elapsed(events[0], events[1]))
      end
      launched
    end

    private def profile_label(node : Node, profile : Profile) : String
      return node.op.to_s unless profile.shapes?
      shapes = node.op.try(&.mat_mul?) ? node.srcs.first(2).map(&.shape) : [node.shape]
      "#{node.op} #{shapes.map { |shape| "[#{shape.join(',')}]" }.join('·')}"
    end

    # Queues the kernel for *node* and returns true, or returns false when it has
    # none: an op not yet ported, a dtype other than `F32`, a view of more than 8
    # dimensions, or a view `Unview` cannot invert.
    private def launch(kernels : Kernels, node : Node, output : CUDABuffer) : Bool
      return false unless node.dtype.f32? && node.srcs.all? { |src| src.ndim <= MAX_DIMS }
      case node.op
      when Op::Sum, Op::Max                           then launch_reduce(kernels, node, output)
      when Op::Cat                                    then launch_cat(kernels, node, output)
      when Op::Write                                  then launch_write(kernels, node, output)
      when Op::Gather, Op::ScatterAdd                 then launch_rows(kernels, node, output)
      when Op::Unview                                 then return launch_unview(kernels, node, output)
      when Op::MatMul                                 then launch_matmul(kernels, node, output)
      when Op::Softmax, Op::LogSoftmax, Op::LayerNorm then launch_row_op(kernels, node, output)
      when Op::Kernel                                 then launch_kernel(node, output)
      else                                                 return launch_elementwise(kernels, node, output)
      end
      true
    end

    # The elementwise primitives: one thread per element. Returns false for an op
    # with no elementwise kernel.
    private def launch_elementwise(kernels : Kernels, node : Node, output : CUDABuffer) : Bool
      srcs = node.srcs
      n = node.size.to_u32
      if srcs.size == 2 && (name = BINARY[node.op]?)
        a, b = srcs
        run(kernels[name], node.size, pointer(a), pointer(b), output.pointer, view_args(a.view), view_args(b.view), n)
      elsif srcs.size == 1 && (name = UNARY[node.op]?)
        a = srcs.first
        run(kernels[name], node.size, pointer(a), output.pointer, view_args(a.view), n)
      else
        return false
      end
      true
    end

    # Sum and Max, folding each output's run of the source permuted so the reduced
    # axis is last: one block per output, with a thread per chunk of the run up
    # to `REDUCE_THREADS`, or one thread per output when the run is one chunk.
    private def launch_reduce(kernels : Kernels, node : Node, output : CUDABuffer) : Nil
      src = node.srcs.first
      view = src.view
      run = view.size
      if axis = node.axis
        view = view.permute((0...view.ndim).reject(axis) + [axis])
        run = view.shape.last
      end
      name = node.op == Op::Sum ? "sum" : "max"
      chunks = (run + REDUCE_CHUNK - 1) // REDUCE_CHUNK
      # A run along the (permuted) last axis, or over a dense or constant view, is
      # walked by stride; one over several dimensions of another view is not.
      walk = (node.axis || view.dense? || view.constant? ? 1 : 0).to_u32
      if chunks <= 1
        run(kernels["#{name}_short_f32"], node.size, pointer(src), output.pointer, view_args(view), run.to_u32, node.size.to_u32, walk)
      elsif node.size > 0
        context.launch(kernels["#{name}_f32"], node.size, Math.min(chunks, REDUCE_THREADS),
          pointer(src), output.pointer, view_args(view), run.to_u32, walk)
      end
    end

    # Cat: one launch per source, copying it into its slice of the output.
    private def launch_cat(kernels : Kernels, node : Node, output : CUDABuffer) : Nil
      axis = node.axis || raise Error.new("cat needs an axis")
      out_view = View.contiguous(node.shape)
      start = 0
      node.srcs.each do |src|
        stop = start + src.shape[axis]
        region = out_view.slice(axis, start, stop)
        run(kernels["copy_into_f32"], src.size, pointer(src), output.pointer, view_args(src.view), view_args(region), src.size.to_u32)
        start = stop
      end
    end

    # Write: one thread per element written, into the target's own buffer.
    private def launch_write(kernels : Kernels, node : Node, output : CUDABuffer) : Nil
      value = node.srcs.last
      region = node.source_view || raise Error.new("write needs a region")
      run(kernels["copy_into_f32"], value.size, pointer(value), output.pointer, view_args(value.view), view_args(region), value.size.to_u32)
    end

    # Gather, one thread per output element, and ScatterAdd, one thread per column.
    # Both report an id out of range through the error buffer (see `#sync`).
    private def launch_rows(kernels : Kernels, node : Node, output : CUDABuffer) : Nil
      rows_src, ids = node.srcs
      rows, width = node.op == Op::Gather ? rows_src.shape : node.shape
      @errors_pending = true
      args = {pointer(rows_src), pointer(ids), output.pointer, errors, view_args(rows_src.view), view_args(ids.view), width.to_u32, rows}
      if node.op == Op::Gather
        run(kernels["gather_f32"], node.size, *args, node.size.to_u32)
      else
        run(kernels["scatter_add_f32"], width, *args, ids.size.to_u32)
      end
    end

    # Unview: one thread per output position, summing what the view reads from it.
    # Returns false for a view whose positions are not each read by a regular grid
    # of indices; see `#inverse_args`.
    private def launch_unview(kernels : Kernels, node : Node, output : CUDABuffer) : Bool
      view = node.source_view || raise Error.new("unview needs a source view")
      return false unless view.ndim <= MAX_DIMS && (inverse = inverse_args(view))
      src = node.srcs.first
      run(kernels["unview_f32"], node.size, pointer(src), output.pointer, view_args(src.view), view_args(view), inverse, node.size.to_u32)
      true
    end

    # MatMul, divided as `#matmul_mode` says.
    #
    # `Split` launches one block per output row and block of `COLS` columns, with
    # `SLICES` threads per column, each summing chunks of `k`. `Tiled` launches
    # one block per `#matmul_tile` outputs of each batch matrix, with a thread per
    # `TILE_PATCH` x `TILE_PATCH` patch. `TwoPass` launches one block per
    # `DECODE_COLS` columns, chunk of `k` and `DECODE_ROWS` rows of each batch
    # matrix, then one thread per output.
    private def launch_matmul(kernels : Kernels, node : Node, output : CUDABuffer) : Nil
      a, b = node.srcs[0], node.srcs[1]
      m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
      return if node.size == 0
      args = {pointer(a), pointer(b), output.pointer, view_args(a.view), view_args(b.view), bias_pointer(node), m.to_u32, k.to_u32, n.to_u32}
      case matmul_mode
      in .serial?
        run(kernels["matmul_f32"], node.size, *args, node.size.to_u32)
      in .split?, .tiled?, .auto?
        if matmul_mode.tiled? || (matmul_mode.auto? && m >= TILED_ROWS)
          rows, cols = matmul_tile.block
          grid = {(n + cols - 1) // cols, (m + rows - 1) // rows, node.size // (m * n)}
          context.launch(kernels[matmul_tile.kernel], grid, {cols // TILE_PATCH, rows // TILE_PATCH}, *args)
        elsif matmul_mode.split? || b.view.strides.last == 1
          context.launch(kernels["matmul_split_f32"], {(n + COLS - 1) // COLS, node.size // n, 1}, {COLS, SLICES}, *args)
        else
          launch_two_pass(kernels, node, output, m, k, n)
        end
      in .two_pass?
        launch_two_pass(kernels, node, output, m, k, n)
      end
    end

    private def launch_two_pass(kernels : Kernels, node : Node, output : CUDABuffer, m : Int32, k : Int32, n : Int32) : Nil
      a, b = node.srcs[0], node.srcs[1]
      chunks = (k + CHUNK - 1) // CHUNK
      scratch = scratch(node.size * chunks * 4)
      if chunks > 0
        grid = {(n + DECODE_COLS - 1) // DECODE_COLS, chunks, node.size // (m * n) * ((m + DECODE_ROWS - 1) // DECODE_ROWS)}
        context.launch(kernels["matmul_chunks_f32"], grid, {DECODE_COLS, 1}, pointer(a), pointer(b), scratch,
          view_args(a.view), view_args(b.view), m.to_u32, k.to_u32, n.to_u32)
      end
      run(kernels["matmul_chunk_sums_f32"], node.size, scratch, output.pointer, bias_pointer(node), m.to_u32, k.to_u32, n.to_u32, node.size.to_u32)
    end

    # Returns a `MatMul`'s bias pointer, or a null pointer when it has none.
    private def bias_pointer(node : Node) : LibCUDA::DevicePtr
      (bias = node.srcs[2]?) ? pointer(bias) : LibCUDA::DevicePtr.new(0)
    end

    # Returns device memory of at least *bytesize* bytes for a kernel's partial
    # results, kept between launches: kernels run one after another, so the next
    # launch may reuse it.
    private def scratch(bytesize : Int32) : LibCUDA::DevicePtr
      current = @scratch
      if current.nil? || current.bytesize < bytesize
        current = alloc(Math.max(bytesize, 4)).as(CUDABuffer)
        @scratch = current
      end
      current.pointer
    end

    # Softmax, LogSoftmax and LayerNorm: one block per row along the last axis,
    # with a thread per chunk of the row up to `REDUCE_THREADS`.
    private def launch_row_op(kernels : Kernels, node : Node, output : CUDABuffer) : Nil
      n = node.shape.last
      rows = node.size // n
      return if rows == 0
      threads = Math.min((n + REDUCE_CHUNK - 1) // REDUCE_CHUNK, REDUCE_THREADS)
      src = node.srcs.first
      if node.op == Op::LayerNorm
        weight, bias = node.srcs[1], node.srcs[2]
        epsilon = node.epsilon || raise Error.new("layernorm needs an epsilon")
        context.launch(kernels["layernorm_f32"], rows, threads, pointer(src), pointer(weight), pointer(bias), output.pointer,
          view_args(src.view), view_args(weight.view), view_args(bias.view), n.to_u32, epsilon.to_f32)
      else
        name = node.op == Op::Softmax ? "softmax_f32" : "log_softmax_f32"
        context.launch(kernels[name], rows, threads, pointer(src), output.pointer, view_args(src.view), n.to_u32)
      end
    end

    # Kernel: one thread per element, running the kernel's generated CUDA C++,
    # compiled the first time the kernel runs on this device. Up to four inputs,
    # which is more than any kernel tensomite defines.
    private def launch_kernel(node : Node, output : CUDABuffer) : Nil
      kernel = node.kernel || raise Error.new("kernel node without a kernel")
      compiled = @macro_kernels[kernel.cuda_name] ||= context.load(CUDA.compile(PRELUDE + kernel.cuda, CUDA.target_arch(gpu), "#{kernel.cuda_name}.cu"))
      function = compiled[kernel.cuda_name]
      s = node.srcs
      n = node.size.to_u32
      case s.size
      when 1 then run(function, node.size, pointer(s[0]), output.pointer, view_args(s[0].view), n)
      when 2 then run(function, node.size, pointer(s[0]), pointer(s[1]), output.pointer, view_args(s[0].view), view_args(s[1].view), n)
      when 3
        run(function, node.size, pointer(s[0]), pointer(s[1]), pointer(s[2]), output.pointer,
          view_args(s[0].view), view_args(s[1].view), view_args(s[2].view), n)
      when 4
        run(function, node.size, pointer(s[0]), pointer(s[1]), pointer(s[2]), pointer(s[3]), output.pointer,
          view_args(s[0].view), view_args(s[1].view), view_args(s[2].view), view_args(s[3].view), n)
      else
        raise DeviceError.new("CUDA kernels take at most 4 inputs, not #{s.size}")
      end
    end

    # Returns the kernels' `Inverse` for *view*: the dimensions with a nonzero
    # stride and more than one index, largest stride first, then the rest.
    #
    # Returns `nil` unless each of those strides exceeds the furthest the smaller
    # ones can reach together, which makes the decomposition of a position unique.
    # Every view tensomite builds satisfies it; overlapping views run on the host.
    private def inverse_args(view : View) : InverseArgs?
      dims = (0...view.ndim)
      strided = dims.select { |dim| view.strides[dim] != 0 && view.shape[dim] > 1 }.sort_by! { |dim| -view.strides[dim] }
      reach = 0
      strided.reverse_each do |dim|
        stride = view.strides[dim]
        return unless stride > reach
        reach += stride * (view.shape[dim] - 1)
      end
      repeated = dims.reject { |dim| strided.includes?(dim) }
      args = InverseArgs.new(0)
      args[0] = strided.size
      strided.each_with_index { |dim, k| args[1 + k] = dim }
      args[1 + MAX_DIMS] = repeated.size
      repeated.each_with_index { |dim, k| args[2 + MAX_DIMS + k] = dim }
      args
    end

    # Queues *function* over *threads* threads, in blocks of `BLOCK`.
    private def run(function : Void*, threads : Int32, *args) : Nil
      return if threads == 0
      context.launch(function, (threads + BLOCK - 1) // BLOCK, BLOCK, *args)
    end

    private def view_args(view : View) : ViewArgs
      args = ViewArgs.new(0)
      args[0], args[1] = view.ndim, view.offset
      view.ndim.times do |dim|
        args[2 + dim] = view.shape[dim]
        args[2 + MAX_DIMS + dim] = view.strides[dim]
      end
      args[2 + 2 * MAX_DIMS] = view.dense? ? 1 : 0
      args
    end

    private def pointer(tensor : Tensor) : LibCUDA::DevicePtr
      cuda(tensor.node.buffer!).pointer
    end

    # The error buffer that `gather_f32` and `scatter_add_f32` report into:
    # a count, then the first id out of range and the number of rows.
    private def errors : LibCUDA::DevicePtr
      @errors ||= context.alloc(12).tap { |pointer| context.upload(pointer, Bytes.new(12)) }
    end

    # Raises `ExecutionError` for the first id out of range a kernel reported
    # since the last check, and clears the report.
    private def check_errors : Nil
      return unless @errors_pending && (pointer = @errors)
      @errors_pending = false
      reports = Slice(Int32).new(3)
      context.download(pointer, reports.to_unsafe_bytes)
      return if reports[0] == 0
      context.upload(pointer, Bytes.new(12))
      raise ExecutionError.new("index #{reports[1]} out of range for #{reports[2]} rows")
    end

    # Returns whether the driver has started and found at least one GPU.
    def self.available? : Bool
      init && gpu_count > 0
    end

    # Returns the CUDA version the driver supports, such as `{13, 2}`.
    def self.driver_version : {Int32, Int32}
      check LibCUDA.cuDriverGetVersion(out version), "cuDriverGetVersion"
      {version // 1000, version % 1000 // 10}
    end

    # Returns every GPU the driver can see.
    def self.gpus : Array(GPU)
      return [] of GPU unless init
      Array.new(gpu_count) do |ordinal|
        check LibCUDA.cuDeviceGet(out device, ordinal), "cuDeviceGet"
        name = Bytes.new(256)
        check LibCUDA.cuDeviceGetName(name.to_unsafe, name.size, device), "cuDeviceGetName"
        check LibCUDA.cuDeviceTotalMem_v2(out memory, device), "cuDeviceTotalMem"
        GPU.new(ordinal, String.new(name.to_unsafe), memory.to_u64,
          {attribute(device, LibCUDA::ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR), attribute(device, LibCUDA::ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)},
          attribute(device, LibCUDA::ATTRIBUTE_MULTIPROCESSOR_COUNT), attribute(device, LibCUDA::ATTRIBUTE_INTEGRATED) != 0)
      end
    end

    # Returns NVRTC's version, such as `{13, 1}`.
    def self.nvrtc_version : {Int32, Int32}
      check_nvrtc LibNVRTC.nvrtcVersion(out major, out minor), "nvrtcVersion"
      {major, minor}
    end

    # Returns the GPU architectures NVRTC can compile for, such as `[75, 80, ...]`.
    def self.supported_archs : Array(Int32)
      check_nvrtc LibNVRTC.nvrtcGetNumSupportedArchs(out count), "nvrtcGetNumSupportedArchs"
      archs = Array(Int32).new(count, 0)
      check_nvrtc LibNVRTC.nvrtcGetSupportedArchs(archs.to_unsafe), "nvrtcGetSupportedArchs"
      archs
    end

    # Returns the architecture to compile for on *gpu*: its own, or the newest
    # older one that NVRTC knows, whose PTX the driver compiles for the GPU.
    def self.target_arch(gpu : GPU) : Int32
      supported_archs.select(&.<=(gpu.arch)).max? ||
        raise DeviceError.new("NVRTC #{nvrtc_version.join('.')} cannot compile for #{gpu.name} (compute #{gpu.compute_capability.join('.')})")
    end

    # Returns the PTX for CUDA C++ *source*, compiled for *arch* with fused
    # multiply-adds off, so `a * b + c` rounds twice, as on the host.
    #
    # Raises `CompileError` with NVRTC's log if the source does not compile.
    def self.compile(source : String, arch : Int32, name : String = "kernels.cu") : String
      check_nvrtc LibNVRTC.nvrtcCreateProgram(out program, source, name, 0, nil, nil), "nvrtcCreateProgram"
      begin
        options = ["--gpu-architecture=compute_#{arch}", "--fmad=false", "--std=c++17"]
        result = LibNVRTC.nvrtcCompileProgram(program, options.size, options.map(&.to_unsafe).to_unsafe)
        unless result == 0
          check_nvrtc LibNVRTC.nvrtcGetProgramLogSize(program, out log_size), "nvrtcGetProgramLogSize"
          log = Bytes.new(log_size)
          check_nvrtc LibNVRTC.nvrtcGetProgramLog(program, log.to_unsafe), "nvrtcGetProgramLog"
          raise CompileError.new("#{name}: #{String.new(log.to_unsafe).strip}")
        end
        check_nvrtc LibNVRTC.nvrtcGetPTXSize(program, out ptx_size), "nvrtcGetPTXSize"
        ptx = Bytes.new(ptx_size)
        check_nvrtc LibNVRTC.nvrtcGetPTX(program, ptx.to_unsafe), "nvrtcGetPTX"
        String.new(ptx.to_unsafe)
      ensure
        LibNVRTC.nvrtcDestroyProgram(pointerof(program))
      end
    end

    # Returns *args* as `cuLaunchKernel` takes them: a pointer to each value, in
    # memory of the value's own type.
    #
    # The loop is unrolled at compile time so that each value keeps its exact
    # type; a block over the tuple would see their union, and box a type tag
    # with each value.
    def self.kernel_params(*args : *T) : Array(Void*) forall T
      params = Array(Void*).new({{ T.size }})
      {% for i in 0...T.size %}
        params << Pointer.malloc(1, args[{{ i }}]).as(Void*)
      {% end %}
      params
    end

    # Raises `DeviceError` naming *call* and the driver's error, unless *result* is success.
    def self.check(result : LibCUDA::Result, call : String) : Nil
      return if result == 0
      LibCUDA.cuGetErrorName(result, out name)
      LibCUDA.cuGetErrorString(result, out text)
      raise DeviceError.new("#{call}: #{name.null? ? result : String.new(name)} (#{text.null? ? "unknown" : String.new(text)})")
    end

    private def self.check_nvrtc(result : LibNVRTC::Result, call : String) : Nil
      raise DeviceError.new("#{call}: #{String.new(LibNVRTC.nvrtcGetErrorString(result))}") unless result == 0
    end

    private def self.init : Bool
      @@initialised = LibCUDA.cuInit(0) == 0 if @@initialised.nil?
      @@initialised.as(Bool)
    end

    private def self.gpu_count : Int32
      LibCUDA.cuDeviceGetCount(out count) == 0 ? count : 0
    end

    private def self.attribute(device : Int32, attribute : Int32) : Int32
      check LibCUDA.cuDeviceGetAttribute(out value, attribute, device), "cuDeviceGetAttribute"
      value
    end

    # A GPU's primary context, current on the calling thread: device memory,
    # loaded kernels and launches. Everything it does is synchronous except
    # `#launch`; `#synchronize` waits for launched kernels.
    #
    # ```
    # context = CUDA::Context.new(CUDA.gpus.first)
    # kernels = context.load(CUDA.compile(source, CUDA.target_arch(context.gpu)))
    # buffer = context.alloc(4 * n)
    # context.launch(kernels["fill"], blocks: (n + 255) // 256, threads: 256, buffer, n)
    # ```
    class Context
      getter gpu : GPU
      @context : Void*

      def initialize(@gpu : GPU)
        CUDA.check LibCUDA.cuDeviceGet(out device, gpu.ordinal), "cuDeviceGet"
        CUDA.check LibCUDA.cuDevicePrimaryCtxRetain(out context, device), "cuDevicePrimaryCtxRetain"
        @context = context
        CUDA.check LibCUDA.cuCtxSetCurrent(@context), "cuCtxSetCurrent"
      end

      # Returns *bytes* of device memory, uninitialised.
      def alloc(bytes : Int) : LibCUDA::DevicePtr
        CUDA.check LibCUDA.cuMemAlloc_v2(out pointer, bytes), "cuMemAlloc"
        pointer
      end

      def free(pointer : LibCUDA::DevicePtr) : Nil
        CUDA.check LibCUDA.cuMemFree_v2(pointer), "cuMemFree"
      end

      # Frees *pointer* from any thread, such as a finalizer's, without raising.
      def free_from_any_thread(pointer : LibCUDA::DevicePtr) : Nil
        LibCUDA.cuCtxSetCurrent(@context)
        LibCUDA.cuMemFree_v2(pointer)
      end

      # Copies *bytes* from the host to device memory at *pointer*.
      def upload(pointer : LibCUDA::DevicePtr, bytes : Bytes) : Nil
        CUDA.check LibCUDA.cuMemcpyHtoD_v2(pointer, bytes.to_unsafe, bytes.size), "cuMemcpyHtoD"
      end

      # Copies device memory at *pointer* into *bytes*, after launched kernels finish.
      def download(pointer : LibCUDA::DevicePtr, bytes : Bytes) : Nil
        CUDA.check LibCUDA.cuMemcpyDtoH_v2(bytes.to_unsafe, pointer, bytes.size), "cuMemcpyDtoH"
      end

      # Loads *ptx*, which the driver compiles for this GPU, and returns its kernels.
      def load(ptx : String) : Kernels
        load?(ptx) || raise DeviceError.new("cuModuleLoadData: CUDA_ERROR_NOT_SUPPORTED")
      end

      # As `#load`, but returns `nil` when the driver does not support loading
      # modules, as the host stub (`scripts/cuda_stub`) does not.
      def load?(ptx : String) : Kernels?
        result = LibCUDA.cuModuleLoadData(out handle, ptx.to_unsafe)
        return if result == NOT_SUPPORTED
        CUDA.check result, "cuModuleLoadData"
        Kernels.new(handle)
      end

      private NOT_SUPPORTED = 801

      # Queues *function* on *blocks* blocks of *threads* threads, with *args*:
      # device pointers (`UInt64`) and numbers, in the kernel's parameter order.
      def launch(function : Void*, blocks : Int32, threads : Int32, *args) : Nil
        launch(function, {blocks, 1, 1}, {threads, 1}, *args)
      end

      # Returns the number of kernels launched since creation.
      getter launches = 0_i64

      # Queues *function* on a grid of *grid* blocks, each of *block* threads.
      def launch(function : Void*, grid : {Int32, Int32, Int32}, block : {Int32, Int32}, *args) : Nil
        return if grid.any?(&.zero?)
        @launches += 1
        params = CUDA.kernel_params(*args)
        CUDA.check LibCUDA.cuLaunchKernel(function, grid[0].to_u32, grid[1].to_u32, grid[2].to_u32,
          block[0].to_u32, block[1].to_u32, 1, 0, Pointer(Void).null, params.to_unsafe, Pointer(Void*).null), "cuLaunchKernel"
      end

      # Waits for every launched kernel, and raises any error one of them hit.
      def synchronize : Nil
        CUDA.check LibCUDA.cuCtxSynchronize, "cuCtxSynchronize"
      end

      # Returns a new event, for timing work on the GPU.
      def event : Void*
        CUDA.check LibCUDA.cuEventCreate(out event, 0), "cuEventCreate"
        event
      end

      # Queues *event* after the work launched so far.
      def record(event : Void*) : Nil
        CUDA.check LibCUDA.cuEventRecord(event, Pointer(Void).null), "cuEventRecord"
      end

      # Waits for *stop*, and returns the GPU seconds between *start* and *stop*.
      def elapsed(start : Void*, stop : Void*) : Float64
        CUDA.check LibCUDA.cuEventSynchronize(stop), "cuEventSynchronize"
        CUDA.check LibCUDA.cuEventElapsedTime(out milliseconds, start, stop), "cuEventElapsedTime"
        milliseconds / 1000.0
      end
    end

    # The kernels of one loaded PTX module, by name.
    class Kernels
      @functions = {} of String => Void*

      def initialize(@handle : Void*)
      end

      # Returns the kernel *name*, declared `extern "C" __global__` in the source.
      def [](name : String) : Void*
        @functions[name] ||= begin
          CUDA.check LibCUDA.cuModuleGetFunction(out function, @handle, name), "cuModuleGetFunction(#{name})"
          function
        end
      end
    end
  end
end

require "./cuda/report"
