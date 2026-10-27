module Tensomite
  # CUDA's driver API, from the NVIDIA driver (`libcuda`). Names ending `_v2` are
  # the symbols the CUDA headers map the unsuffixed names to.
  @[Link("cuda")]
  lib LibCUDA
    alias Result = Int32
    alias DevicePtr = UInt64

    ATTRIBUTE_MULTIPROCESSOR_COUNT     = 16
    ATTRIBUTE_INTEGRATED               = 18
    ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
    ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76

    fun cuInit(flags : UInt32) : Result
    fun cuDriverGetVersion(version : Int32*) : Result
    fun cuGetErrorName(result : Result, name : UInt8**) : Result
    fun cuGetErrorString(result : Result, text : UInt8**) : Result

    fun cuDeviceGetCount(count : Int32*) : Result
    fun cuDeviceGet(device : Int32*, ordinal : Int32) : Result
    fun cuDeviceGetName(name : UInt8*, length : Int32, device : Int32) : Result
    fun cuDeviceTotalMem_v2(bytes : LibC::SizeT*, device : Int32) : Result
    fun cuDeviceGetAttribute(value : Int32*, attribute : Int32, device : Int32) : Result

    fun cuDevicePrimaryCtxRetain(context : Void**, device : Int32) : Result
    fun cuDevicePrimaryCtxRelease_v2(device : Int32) : Result
    fun cuCtxSetCurrent(context : Void*) : Result
    fun cuCtxSynchronize : Result

    fun cuMemAlloc_v2(pointer : DevicePtr*, bytes : LibC::SizeT) : Result
    fun cuMemFree_v2(pointer : DevicePtr) : Result
    fun cuMemcpyHtoD_v2(destination : DevicePtr, source : Void*, bytes : LibC::SizeT) : Result
    fun cuMemcpyDtoH_v2(destination : Void*, source : DevicePtr, bytes : LibC::SizeT) : Result

    fun cuModuleLoadData(module : Void**, image : Void*) : Result
    fun cuModuleUnload(module : Void*) : Result
    fun cuModuleGetFunction(function : Void**, module : Void*, name : UInt8*) : Result
    fun cuEventCreate(event : Void**, flags : UInt32) : Result
    fun cuEventDestroy_v2(event : Void*) : Result
    fun cuEventRecord(event : Void*, stream : Void*) : Result
    fun cuEventSynchronize(event : Void*) : Result
    fun cuEventElapsedTime(milliseconds : Float32*, start : Void*, stop : Void*) : Result

    fun cuLaunchKernel(function : Void*, grid_x : UInt32, grid_y : UInt32, grid_z : UInt32,
                       block_x : UInt32, block_y : UInt32, block_z : UInt32, shared_bytes : UInt32,
                       stream : Void*, params : Void**, extra : Void**) : Result
  end

  # NVRTC, CUDA's runtime compiler (`libnvrtc`), which turns CUDA C++ into PTX.
  @[Link("nvrtc")]
  lib LibNVRTC
    alias Result = Int32

    fun nvrtcVersion(major : Int32*, minor : Int32*) : Result
    fun nvrtcGetErrorString(result : Result) : UInt8*
    fun nvrtcGetNumSupportedArchs(count : Int32*) : Result
    fun nvrtcGetSupportedArchs(archs : Int32*) : Result
    fun nvrtcCreateProgram(program : Void**, source : UInt8*, name : UInt8*, header_count : Int32,
                           headers : UInt8**, include_names : UInt8**) : Result
    fun nvrtcDestroyProgram(program : Void**) : Result
    fun nvrtcCompileProgram(program : Void*, option_count : Int32, options : UInt8**) : Result
    fun nvrtcGetPTXSize(program : Void*, bytes : LibC::SizeT*) : Result
    fun nvrtcGetPTX(program : Void*, ptx : UInt8*) : Result
    fun nvrtcGetProgramLogSize(program : Void*, bytes : LibC::SizeT*) : Result
    fun nvrtcGetProgramLog(program : Void*, log : UInt8*) : Result
  end
end
