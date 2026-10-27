/* A stand-in for libcuda on machines with no NVIDIA driver. It reports one GPU,
 * "tensomite host stub", whose memory is host memory: allocations and copies
 * work, so the CUDA device runs its host kernels and transfers.
 *
 * Loading PTX fails with CUDA_ERROR_NOT_SUPPORTED, unless TENSOMITE_CUDA_KERNELS
 * names a library of kernels compiled as host C++ (emulate.h). Then loading any
 * module succeeds, its kernels are that library's, found by name, and a launch
 * runs them on the host. */
#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int CUresult;
typedef uint64_t CUdeviceptr;

#define SUCCESS 0
#define INVALID_VALUE 1
#define OUT_OF_MEMORY 2
#define NOT_FOUND 500
#define NOT_SUPPORTED 801

static int context;

CUresult cuInit(unsigned flags) { return SUCCESS; }
CUresult cuDriverGetVersion(int *version) { *version = 13020; return SUCCESS; }

CUresult cuGetErrorName(CUresult result, const char **name) {
  *name = result == NOT_SUPPORTED ? "CUDA_ERROR_NOT_SUPPORTED" : result == NOT_FOUND ? "CUDA_ERROR_NOT_FOUND" : result == OUT_OF_MEMORY ? "CUDA_ERROR_OUT_OF_MEMORY" : "CUDA_ERROR_INVALID_VALUE";
  return SUCCESS;
}

CUresult cuGetErrorString(CUresult result, const char **text) {
  *text = result == NOT_SUPPORTED ? "the host stub runs no kernels" : "host stub error";
  return SUCCESS;
}

CUresult cuDeviceGetCount(int *count) { *count = 1; return SUCCESS; }
CUresult cuDeviceGet(int *device, int ordinal) { *device = ordinal; return ordinal == 0 ? SUCCESS : INVALID_VALUE; }

CUresult cuDeviceGetName(char *name, int length, int device) {
  strncpy(name, "tensomite host stub", length);
  return SUCCESS;
}

CUresult cuDeviceTotalMem_v2(size_t *bytes, int device) { *bytes = (size_t)1 << 30; return SUCCESS; }

/* Compute capability 7.5, multiprocessor count, integrated. */
CUresult cuDeviceGetAttribute(int *value, int attribute, int device) {
  switch (attribute) {
  case 75: *value = 7; break;
  case 76: *value = 5; break;
  case 16: *value = 1; break;
  default: *value = 0;
  }
  return SUCCESS;
}

CUresult cuDevicePrimaryCtxRetain(void **ctx, int device) { *ctx = &context; return SUCCESS; }
CUresult cuDevicePrimaryCtxRelease_v2(int device) { return SUCCESS; }
CUresult cuCtxSetCurrent(void *ctx) { return SUCCESS; }
CUresult cuCtxSynchronize(void) { return SUCCESS; }

CUresult cuMemAlloc_v2(CUdeviceptr *pointer, size_t bytes) {
  void *memory = malloc(bytes);
  if (!memory) return OUT_OF_MEMORY;
  *pointer = (CUdeviceptr)(uintptr_t)memory;
  return SUCCESS;
}

CUresult cuMemFree_v2(CUdeviceptr pointer) { free((void *)(uintptr_t)pointer); return SUCCESS; }

CUresult cuMemcpyHtoD_v2(CUdeviceptr destination, const void *source, size_t bytes) {
  memcpy((void *)(uintptr_t)destination, source, bytes);
  return SUCCESS;
}

CUresult cuMemcpyDtoH_v2(void *destination, CUdeviceptr source, size_t bytes) {
  memcpy(destination, (const void *)(uintptr_t)source, bytes);
  return SUCCESS;
}

static void *(*find_kernel)(const char *);
static void (*launch_kernel)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, void **);

CUresult cuModuleLoadData(void **module, const void *image) {
  const char *path = getenv("TENSOMITE_CUDA_KERNELS");
  void *library = path ? dlopen(path, RTLD_NOW) : NULL;
  if (!library) return NOT_SUPPORTED;
  find_kernel = (void *(*)(const char *))dlsym(library, "tensomite_kernel");
  launch_kernel = (void (*)(void *, unsigned, unsigned, unsigned, unsigned, unsigned, void **))dlsym(library, "tensomite_launch");
  if (!find_kernel || !launch_kernel) return NOT_SUPPORTED;
  *module = library;
  return SUCCESS;
}

CUresult cuModuleUnload(void *module) { return SUCCESS; }

CUresult cuModuleGetFunction(void **function, void *module, const char *name) {
  *function = find_kernel(name);
  return *function ? SUCCESS : NOT_FOUND;
}

/* Runs grids of up to three dimensions of blocks of up to two. */
CUresult cuLaunchKernel(void *function, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                        unsigned bz, unsigned shared, void *stream, void **params, void **extra) {
  if (bz != 1) return NOT_SUPPORTED;
  launch_kernel(function, gx, gy, gz, bx, by, params);
  return SUCCESS;
}
