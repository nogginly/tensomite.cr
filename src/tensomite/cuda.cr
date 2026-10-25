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
  # NVIDIA GPUs through CUDA's driver API, with kernels compiled from CUDA C++ by
  # NVRTC when they are first needed.
  class CUDA
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
        CUDA.check LibCUDA.cuModuleLoadData(out handle, ptx.to_unsafe), "cuModuleLoadData"
        Kernels.new(handle)
      end

      # Queues *function* on *blocks* blocks of *threads* threads, with *args*:
      # device pointers (`UInt64`) and numbers, in the kernel's parameter order.
      def launch(function : Void*, blocks : Int32, threads : Int32, *args) : Nil
        params = args.map { |arg| Pointer.malloc(1, arg).as(Void*) }.to_a
        CUDA.check LibCUDA.cuLaunchKernel(function, blocks.to_u32, 1, 1, threads.to_u32, 1, 1, 0,
          Pointer(Void).null, params.to_unsafe, Pointer(Void*).null), "cuLaunchKernel"
      end

      # Waits for every launched kernel, and raises any error one of them hit.
      def synchronize : Nil
        CUDA.check LibCUDA.cuCtxSynchronize, "cuCtxSynchronize"
      end
    end

    # The kernels of one loaded PTX module, by name.
    class Kernels
      def initialize(@handle : Void*)
      end

      # Returns the kernel *name*, declared `extern "C" __global__` in the source.
      def [](name : String) : Void*
        CUDA.check LibCUDA.cuModuleGetFunction(out function, @handle, name), "cuModuleGetFunction(#{name})"
        function
      end
    end
  end
end
