// Compiles CUDA kernels as host C++, for the stand-in libcuda: each kernel becomes
// a function, and `tensomite_launch` runs a grid of up to three dimensions of
// blocks of up to two, one block after another. Kernels that call __syncthreads
// run their threads as OS threads, meeting at a std::barrier; the others run
// their threads in a loop. Shared memory is a static array, which the threads of
// the one running block share.
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

// Warp shuffles: each thread of the block posts its value, all meet, each
// reads the value of the lane it asks for within its warp, and all meet again
// before the slots are reused. Every thread of the block must call them the
// same number of times, as the kernels that use them ensure.
static unsigned int shuffle_slots[1024];
template <class T>
static T shuffle_from(T value, unsigned int source) {
    static_assert(sizeof(T) == 4, "shuffles move 4-byte values");
    unsigned int t = threadIdx.y * blockDim.x + threadIdx.x;
    std::memcpy(&shuffle_slots[t], &value, 4);
    block_barrier->arrive_and_wait();
    T other;
    std::memcpy(&other, &shuffle_slots[source], 4);
    block_barrier->arrive_and_wait();
    return other;
}
template <class T>
static T __shfl_sync(unsigned int, T value, int lane) {
    unsigned int t = threadIdx.y * blockDim.x + threadIdx.x;
    return shuffle_from(value, t / 32 * 32 + unsigned(lane % 32));
}
template <class T>
static T __shfl_xor_sync(unsigned int, T value, int mask) {
    unsigned int t = threadIdx.y * blockDim.x + threadIdx.x;
    return shuffle_from(value, t ^ unsigned(mask));
}

// A warp's barrier: here the whole block's, which every kernel that calls it
// reaches from every thread the same number of times.
#define __syncwarp() block_barrier->arrive_and_wait()

// The dot product of the four signed bytes of a with those of b, plus c.
static int __dp4a(int a, int b, int c) {
    for (int j = 0; j < 4; j++) c += int((signed char)(a >> (8 * j))) * int((signed char)(b >> (8 * j)));
    return c;
}

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

extern "C" void tensomite_launch(void* handle, unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
                                 unsigned int block_x, unsigned int block_y, void** params) {
    Kernel* kernel = static_cast<Kernel*>(handle);
    blockDim = dim3{block_x, block_y, 1};
    unsigned int threads = block_x * block_y;
    unsigned int blocks = grid_x * grid_y * grid_z;
    auto block_of = [=](unsigned int b) { return dim3{b % grid_x, b / grid_x % grid_y, b / (grid_x * grid_y)}; };
    if (!kernel->sync) {
        for (unsigned int b = 0; b < blocks; b++) {
            blockIdx = block_of(b);
            for (unsigned int t = 0; t < threads; t++) {
                threadIdx = dim3{t % block_x, t / block_x, 0};
                kernel->call(params);
            }
        }
        return;
    }
    // One OS thread per GPU thread for the whole grid: each runs its thread of
    // every block in turn, and all meet at the barrier before the next block, so
    // shared memory is not reused while a thread still reads it.
    std::barrier<> barrier(threads);
    block_barrier = &barrier;
    std::vector<std::thread> workers;
    for (unsigned int t = 0; t < threads; t++)
        workers.emplace_back([=, &barrier] {
            threadIdx = dim3{t % block_x, t / block_x, 0};
            for (unsigned int b = 0; b < blocks; b++) {
                blockIdx = block_of(b);
                kernel->call(params);
                barrier.arrive_and_wait();
            }
        });
    for (auto& worker : workers) worker.join();
}
