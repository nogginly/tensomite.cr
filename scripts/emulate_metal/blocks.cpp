// Runs the Metal block-format kernels (quantize_rows_q8_f32, block_matmul_f32
// and gather_blocks_f32) on the CPU, and checks their results against the
// host's, bit for bit: block_matmul_lanes_f32 as threadgroups of SIMD groups,
// with 1 to 8 outputs a group, the others one thread at a time. Reads the file blocks_data.cr writes.
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
#include "shim.h"
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef _Float16 half;
typedef std::atomic<int> atomic_int;
using std::fabs;
using std::max;
using std::round;
enum memory_order { memory_order_relaxed };
inline int atomic_fetch_add_explicit(atomic_int* a, int v, memory_order) { return a->fetch_add(v); }
inline void atomic_store_explicit(atomic_int* a, int v, memory_order) { a->store(v); }
template <typename T, typename U> T as_type(U value) {
    T result;
    std::memcpy(&result, &value, sizeof(T));
    return result;
}
#include "blocks.inc"

// Runs block_matmul_lanes_f32 over `groups` threadgroups of `simds` SIMD groups, each
// threadgroup's threads as OS threads meeting at a barrier.
template <class... A>
static void run_simd_groups(unsigned int groups, unsigned int simds, A... args) {
    for (unsigned int group = 0; group < groups; group++) {
        std::barrier<> barrier(simds * 32);
        current_barrier = &barrier;
        std::vector<std::thread> threads;
        for (unsigned int t = 0; t < simds * 32; t++)
            threads.emplace_back([=]() mutable {
                block_matmul_lanes_f32(args..., group, t / 32, simds, t % 32);
            });
        for (auto& thread : threads) thread.join();
    }
}

static FILE* in;
static uint read_u32() { uint v; if (fread(&v, 4, 1, in) != 1) throw 1; return v; }
static std::vector<float> read_floats() {
    std::vector<float> v(read_u32());
    if (fread(v.data(), 4, v.size(), in) != v.size()) throw 1;
    return v;
}
static std::vector<uchar> read_bytes() {
    std::vector<uchar> v(read_u32());
    if (fread(v.data(), 1, v.size(), in) != v.size()) throw 1;
    return v;
}

static int compare(const char* what, const std::vector<float>& got, const std::vector<float>& want) {
    for (size_t i = 0; i < want.size(); i++) {
        if (std::memcmp(&got[i], &want[i], 4) != 0) {
            printf("%s: element %zu is %.9g, the host's %.9g\n", what, i, got[i], want[i]);
            return 1;
        }
    }
    printf("%s: %zu values, the host's bits\n", what, want.size());
    return 0;
}

int main(int argc, char** argv) {
    in = fopen(argv[1], "rb");
    int failures = 0;
    const char* names[] = {"Q8_0", "Q4_0", "Q4_1"};
    const uint bytes[] = {34, 18, 20};
    for (int f = 0; f < 3; f++) {
        uint kind = read_u32(), rows = read_u32(), n = read_u32(), k = read_u32();
        auto x = read_floats();
        auto w = read_bytes();
        auto bias = read_floats();
        auto want = read_floats();
        auto gathered = read_floats();
        View vx{};
        vx.ndim = 1; vx.shape[0] = rows * k; vx.strides[0] = 1; vx.dense = 1;
        uint blocks = rows * k / 32;
        std::vector<float> scales(blocks), sums(blocks);
        std::vector<char> qs(rows * k);
        for (uint i = 0; i < blocks; i++) quantize_rows_q8_f32(x.data(), scales.data(), sums.data(), qs.data(), vx, blocks, i);
        Blocks b{kind, bytes[kind], 0};
        uint3 dims{n, k, 1};
        uint count = rows * n;
        std::vector<float> got(count);
        char label[64];
        for (uint i = 0; i < count; i++)
            block_matmul_f32(w.data(), scales.data(), sums.data(), qs.data(), got.data(), bias.data(), b, dims, count, i);
        snprintf(label, sizeof label, "block_matmul %s", names[f]);
        failures += compare(label, got, want);
        for (uint per : {1u, 2u, 4u, 7u, 8u}) {
            std::fill(got.begin(), got.end(), 0.0f);
            run_simd_groups((count + 4 * per - 1) / (4 * per), 4, w.data(), scales.data(), sums.data(), qs.data(), got.data(),
                            bias.data(), b, dims, count, per);
            snprintf(label, sizeof label, "block_matmul_lanes %s, %u per group", names[f], per);
            failures += compare(label, got, want);
        }
        int ids[] = {3, 0, 36, 3};
        View vi{};
        vi.ndim = 1; vi.shape[0] = 4; vi.strides[0] = 1; vi.dense = 1;
        atomic_int errors[3] = {0, 0, 0};
        uint width = k, total = 4 * k;
        int table_rows = n;
        std::vector<float> rows_out(total);
        for (uint i = 0; i < total; i++)
            gather_blocks_f32(w.data(), ids, rows_out.data(), errors, b, vi, width, table_rows, total, i);
        snprintf(label, sizeof label, "gather_blocks %s", names[f]);
        failures += compare(label, rows_out, gathered);
    }
    return failures;
}
