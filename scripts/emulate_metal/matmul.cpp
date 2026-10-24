// Checks that the tiled Metal matmul gives split-K's bits, emulated on the CPU.
// Built and run by run.sh, which extracts the kernels into kernels.inc.
//
// Each case runs matmul_split_f32, every matmul_tiled variant and, as a control,
// the serial matmul_f32, which adds over k in plain index order and so should
// differ from the others whenever k is more than one chunk of 64.
#include "shim.h"
#include "kernels.inc"
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <string>

// Runs one launch: every threadgroup in turn, its threads concurrently.
template <typename F>
void launch(uint3 groups, uint2 threads, F body) {
  for (uint z = 0; z < groups.z; z++)
    for (uint y = 0; y < groups.y; y++)
      for (uint x = 0; x < groups.x; x++) {
        std::barrier<> bar(threads.x * threads.y);
        current_barrier = &bar;
        std::vector<std::thread> pool;
        for (uint ty = 0; ty < threads.y; ty++)
          for (uint tx = 0; tx < threads.x; tx++)
            pool.emplace_back([&, tx, ty] { body(uint3{x, y, z}, uint2{tx, ty}); });
        for (auto& t : pool) t.join();
      }
}

View make_view(std::vector<int> shape, std::vector<int> strides, int offset) {
  View v{}; v.ndim = shape.size(); v.offset = offset;
  bool dense = true; int expect = 1;
  for (int d = v.ndim - 1; d >= 0; d--) { v.shape[d] = shape[d]; v.strides[d] = strides[d]; if (strides[d] != expect) dense = false; expect *= shape[d]; }
  v.dense = dense; return v;
}

static uint bits(float f) { uint u; memcpy(&u, &f, 4); return u; }

int main() {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> uni(-1, 1);
  int failures = 0, cases = 0;
  struct Case { uint batch, m, k, n; bool ta, tb; };
  std::vector<Case> list = {{1,1,1,1,0,0},{1,3,200,70,0,0},{1,70,63,33,0,0},{1,64,64,64,0,0},{2,37,129,65,0,0},
    {3,17,577,40,1,0},{1,33,100,97,0,1},{2,65,700,31,1,1},{1,128,768,96,0,1},{1,5,1,3,0,0},{1,100,512,100,0,0},{4,20,64,20,1,0}};
  for (auto c : list) {
    // a is [batch, m, k] (stored transposed per matrix when ta), b is [batch, k, n] (transposed when tb).
    std::vector<float> a(c.batch * c.m * c.k), b(c.batch * c.k * c.n);
    for (auto& v : a) v = uni(rng);
    for (auto& v : b) v = uni(rng);
    if (c.k == 1) { a[0] = -0.0f; b[0] = 0.5f; }  // a -0 total for the first output
    View va = c.ta ? make_view({(int)c.batch,(int)c.m,(int)c.k},{(int)(c.m*c.k),1,(int)c.m},0)
                   : make_view({(int)c.batch,(int)c.m,(int)c.k},{(int)(c.m*c.k),(int)c.k,1},0);
    View vb = c.tb ? make_view({(int)c.batch,(int)c.k,(int)c.n},{(int)(c.k*c.n),1,(int)c.k},0)
                   : make_view({(int)c.batch,(int)c.k,(int)c.n},{(int)(c.k*c.n),(int)c.n,1},0);
    uint dims[3] = {c.m, c.k, c.n};
    uint rows = c.batch * c.m;
    std::vector<float> split(rows * c.n), t32(rows * c.n), t64(rows * c.n), t3264(rows * c.n);
    launch({(c.n + COLS - 1) / COLS, rows, 1}, {COLS, SLICES}, [&](uint3 g, uint2 t) {
      matmul_split_f32(a.data(), b.data(), split.data(), va, vb, dims, uint2{g.x, g.y}, t); });
    launch({(c.n + 31) / 32, (c.m + 31) / 32, c.batch}, {8, 8}, [&](uint3 g, uint2 t) {
      matmul_tiled<32, 32, 64, 4, 4>(a.data(), b.data(), t32.data(), va, vb, dims, g, uint3{t.x, t.y, 0}); });
    launch({(c.n + 63) / 64, (c.m + 63) / 64, c.batch}, {16, 16}, [&](uint3 g, uint2 t) {
      matmul_tiled<64, 64, 32, 4, 4>(a.data(), b.data(), t64.data(), va, vb, dims, g, uint3{t.x, t.y, 0}); });
    launch({(c.n + 63) / 64, (c.m + 31) / 32, c.batch}, {16, 8}, [&](uint3 g, uint2 t) {
      matmul_tiled<32, 64, 32, 4, 4>(a.data(), b.data(), t3264.data(), va, vb, dims, g, uint3{t.x, t.y, 0}); });
    std::vector<float> serial(rows * c.n);
    uint count = rows * c.n;
    launch({rows * c.n, 1, 1}, {1, 1}, [&](uint3 g, uint2) {
      matmul_f32(a.data(), b.data(), serial.data(), va, vb, dims, count, g.x); });
    int differ = 0;
    for (size_t i = 0; i < split.size(); i++) differ += bits(serial[i]) != bits(t32[i]);
    printf("  control: serial differs from tiled in %d of %zu outputs\n", differ, split.size());
    int bad = 0;
    for (size_t i = 0; i < split.size(); i++)
      if (bits(split[i]) != bits(t32[i]) || bits(split[i]) != bits(t64[i]) || bits(split[i]) != bits(t3264[i])) {
        if (bad++ < 3) printf("  mismatch at %zu: split %08x t32 %08x t64 %08x t32x64 %08x\n", i, bits(split[i]), bits(t32[i]), bits(t64[i]), bits(t3264[i]));
      }
    cases++;
    printf("%s batch %u m %u k %u n %u%s%s\n", bad ? "FAIL" : "ok  ", c.batch, c.m, c.k, c.n, c.ta ? " a^T" : "", c.tb ? " b^T" : "");
    if (bad) failures++;
  }
  printf("%d of %d cases bit-identical\n", cases - failures, cases);
  return failures != 0;
}
