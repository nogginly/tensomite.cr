// Compiles CUDA kernels as host C++, for the stand-in libcuda: each kernel becomes
// a function, and `tensomite_launch` runs a grid of blocks one after another.
// Kernels that call __syncthreads run each block's threads as OS threads, meeting
// at a std::barrier; the others run their threads in a loop. Shared memory is a
// static array, which the threads of the one running block share.
#include <algorithm>
#include <barrier>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using std::max;
using std::min;

struct dim3 {
    unsigned int x = 0, y = 0, z = 0;
};

static thread_local dim3 blockIdx, threadIdx;
static dim3 blockDim;
static std::barrier<>* block_barrier = nullptr;

#define __global__
#define __device__ static
#define __shared__ static
#define __syncthreads() block_barrier->arrive_and_wait()

static int atomicAdd(int* address, int value) { return __atomic_fetch_add(address, value, __ATOMIC_RELAXED); }

struct Kernel {
    void (*call)(void**);
    bool sync;
};

static std::map<std::string, Kernel>& registry() {
    static std::map<std::string, Kernel> kernels;
    return kernels;
}

template <class... A, size_t... I>
static void invoke(void (*f)(A...), void** params, std::index_sequence<I...>) {
    f(*static_cast<std::remove_cv_t<std::remove_reference_t<A>>*>(params[I])...);
}

template <class... A>
static void invoke(void (*f)(A...), void** params) {
    invoke(f, params, std::index_sequence_for<A...>{});
}

#define REGISTER(name, sync)                                                         \
    static bool registered_##name = (registry()[#name] = Kernel{                     \
        [](void** params) { invoke(name, params); }, sync}, true);

extern "C" void* tensomite_kernel(const char* name) {
    auto found = registry().find(name);
    return found == registry().end() ? nullptr : &found->second;
}

extern "C" void tensomite_launch(void* handle, unsigned int blocks, unsigned int threads, void** params) {
    Kernel* kernel = static_cast<Kernel*>(handle);
    blockDim = dim3{threads, 1, 1};
    for (unsigned int b = 0; b < blocks; b++) {
        if (kernel->sync) {
            std::barrier<> barrier(threads);
            block_barrier = &barrier;
            std::vector<std::thread> workers;
            for (unsigned int t = 0; t < threads; t++)
                workers.emplace_back([=] {
                    blockIdx = dim3{b, 0, 0};
                    threadIdx = dim3{t, 0, 0};
                    kernel->call(params);
                });
            for (auto& worker : workers) worker.join();
        } else {
            blockIdx = dim3{b, 0, 0};
            for (unsigned int t = 0; t < threads; t++) {
                threadIdx = dim3{t, 0, 0};
                kernel->call(params);
            }
        }
    }
}
