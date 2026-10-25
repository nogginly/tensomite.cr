/* A stand-in for libcuda with no GPU: every call fails with CUDA_ERROR_NO_DEVICE. */
#define F(name) int name() { return 100; }
F(cuInit) F(cuDriverGetVersion) F(cuGetErrorName) F(cuGetErrorString) F(cuDeviceGetCount)
F(cuDeviceGet) F(cuDeviceGetName) F(cuDeviceTotalMem_v2) F(cuDeviceGetAttribute)
F(cuDevicePrimaryCtxRetain) F(cuDevicePrimaryCtxRelease_v2) F(cuCtxSetCurrent) F(cuCtxSynchronize)
F(cuMemAlloc_v2) F(cuMemFree_v2) F(cuMemcpyHtoD_v2) F(cuMemcpyDtoH_v2) F(cuModuleLoadData)
F(cuModuleUnload) F(cuModuleGetFunction) F(cuLaunchKernel)
