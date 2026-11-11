// Just enough of MSL for kernels to compile as C++ and run on the CPU.
//
// Address-space qualifiers vanish, except that threadgroup variables become
// static, so the threads of one threadgroup share them; run one threadgroup at a
// time. threadgroup_barrier waits on `current_barrier`, which the launcher sets
// to a barrier for that threadgroup's threads.
#include <cstdint>
#include <algorithm>
#include <barrier>
#include <thread>
#include <functional>
typedef uint32_t uint;
struct uint2 { uint x, y; };
struct uint3 { uint x, y, z; };
#define kernel
#define device
#define constant
#define threadgroup static
#define host_name(x)
using std::min;
inline std::barrier<>* current_barrier;
enum class mem_flags { mem_threadgroup };
inline void threadgroup_barrier(mem_flags) { current_barrier->arrive_and_wait(); }
