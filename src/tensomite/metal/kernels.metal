// Hand-written kernels for the Metal device, compiled after prelude.metal with
// fast math off, so +, -, * and comparisons give the same bits as the host
// kernels.

#define BINARY(name, expr)                                                     \
    kernel void name(device const float* a [[buffer(0)]],                     \
                     device const float* b [[buffer(1)]],                     \
                     device float* dst [[buffer(2)]],                         \
                     constant View& va [[buffer(3)]],                         \
                     constant View& vb [[buffer(4)]],                         \
                     constant uint& n [[buffer(5)]],                          \
                     uint i [[thread_position_in_grid]]) {                    \
        if (i >= n) return;                                                    \
        float x = a[offset_of(va, i)];                                         \
        float y = b[offset_of(vb, i)];                                         \
        dst[i] = (expr);                                                       \
    }

#define UNARY(name, expr)                                                      \
    kernel void name(device const float* a [[buffer(0)]],                     \
                     device float* dst [[buffer(1)]],                         \
                     constant View& va [[buffer(2)]],                         \
                     constant uint& n [[buffer(3)]],                          \
                     uint i [[thread_position_in_grid]]) {                    \
        if (i >= n) return;                                                    \
        float x = a[offset_of(va, i)];                                         \
        dst[i] = (expr);                                                       \
    }

BINARY(add_f32, x + y)
BINARY(sub_f32, x - y)
BINARY(mul_f32, x * y)
BINARY(div_f32, x / y)
BINARY(maximum_f32, x > y ? x : y)
BINARY(greater_equal_f32, x >= y ? 1.0f : 0.0f)

UNARY(exp_f32, precise::exp(x))
UNARY(log_f32, precise::log(x))
UNARY(tanh_f32, precise::tanh(x))
UNARY(sqrt_f32, precise::sqrt(x))
UNARY(copy_f32, x)

// Copies source element i into output position offset_of(vd, i): Copy into a
// region of the output, as Cat does for each source.
kernel void copy_into_f32(device const float* src [[buffer(0)]],
                          device float* dst [[buffer(1)]],
                          constant View& vs [[buffer(2)]],
                          constant View& vd [[buffer(3)]],
                          constant uint& n [[buffer(4)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    dst[offset_of(vd, i)] = src[offset_of(vs, i)];
}

// Reductions fold each output's run of `run` consecutive logical indices of `v`.
//
// The run is cut into chunks of REDUCE_CHUNK at fixed offsets. Each thread of an
// output's threadgroup folds whole chunks in index order; the first thread then
// folds the chunk results in chunk order. Chunk boundaries never depend on the
// run's length or on the number of threads, so a run padded with zeros (Sum) or
// with -inf (Max), as masked attention rows are, gives the same bits. Max is
// exact in any order, so it matches the host bit for bit; Sum adds in a different
// order from the host's (Decision X).
//
// When `walk` is set, the run is one stretch of the view's last dimension, and
// its elements are found from its start by the last stride; otherwise each goes
// through offset_of.
//
// A run of one chunk or less is folded by one thread per output (sum_short_f32,
// max_short_f32), which is the same arithmetic as a single chunk.
#define REDUCE_THREADS 256
#define REDUCE_CHUNK 16

// Returns the position of element s of the run that starts at logical index base.
static int run_position(constant View& v, int start, uint walk, uint base, uint s) {
    return walk ? start + int(s) * run_stride(v) : offset_of(v, base + s);
}

kernel void sum_short_f32(device const float* a [[buffer(0)]],
                          device float* dst [[buffer(1)]],
                          constant View& v [[buffer(2)]],
                          constant uint& run [[buffer(3)]],
                          constant uint& count [[buffer(4)]],
                          constant uint& walk [[buffer(5)]],
                          uint row [[thread_position_in_grid]]) {
    if (row >= count) return;
    int start = offset_of(v, row * run);
    float acc = 0.0f;
    for (uint j = 0; j < run; j++) acc = acc + a[run_position(v, start, walk, row * run, j)];
    dst[row] = 0.0f + acc;
}

kernel void max_short_f32(device const float* a [[buffer(0)]],
                          device float* dst [[buffer(1)]],
                          constant View& v [[buffer(2)]],
                          constant uint& run [[buffer(3)]],
                          constant uint& count [[buffer(4)]],
                          constant uint& walk [[buffer(5)]],
                          uint row [[thread_position_in_grid]]) {
    if (row >= count) return;
    int start = offset_of(v, row * run);
    float acc = a[start];
    for (uint j = 1; j < run; j++) {
        float x = a[run_position(v, start, walk, row * run, j)];
        acc = x > acc ? x : acc;
    }
    dst[row] = acc;
}

kernel void sum_f32(device const float* a [[buffer(0)]],
                    device float* dst [[buffer(1)]],
                    constant View& v [[buffer(2)]],
                    constant uint& run [[buffer(3)]],
                    constant uint& walk [[buffer(4)]],
                    uint row [[threadgroup_position_in_grid]],
                    uint t [[thread_position_in_threadgroup]],
                    uint threads [[threads_per_threadgroup]]) {
    threadgroup float partial[REDUCE_THREADS];
    uint base = row * run;
    int start = offset_of(v, base);
    uint chunks = (run + REDUCE_CHUNK - 1) / REDUCE_CHUNK;
    float total = 0.0f;
    for (uint first = 0; first < chunks; first += threads) {
        uint c = first + t;
        float acc = 0.0f;
        if (c < chunks) {
            uint stop = min(run, (c + 1) * REDUCE_CHUNK);
            for (uint s = c * REDUCE_CHUNK; s < stop; s++) acc = acc + a[run_position(v, start, walk, base, s)];
        }
        partial[t] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            uint n = min(threads, chunks - first);
            for (uint k = 0; k < n; k++) total = total + partial[k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0) dst[row] = total;
}

kernel void max_f32(device const float* a [[buffer(0)]],
                    device float* dst [[buffer(1)]],
                    constant View& v [[buffer(2)]],
                    constant uint& run [[buffer(3)]],
                    constant uint& walk [[buffer(4)]],
                    uint row [[threadgroup_position_in_grid]],
                    uint t [[thread_position_in_threadgroup]],
                    uint threads [[threads_per_threadgroup]]) {
    threadgroup float partial[REDUCE_THREADS];
    uint base = row * run;
    int start = offset_of(v, base);
    uint chunks = (run + REDUCE_CHUNK - 1) / REDUCE_CHUNK;
    float total = a[start];
    for (uint first = 0; first < chunks; first += threads) {
        uint c = first + t;
        if (c < chunks) {
            uint stop = min(run, (c + 1) * REDUCE_CHUNK);
            float acc = a[run_position(v, start, walk, base, c * REDUCE_CHUNK)];
            for (uint s = c * REDUCE_CHUNK + 1; s < stop; s++) {
                float x = a[run_position(v, start, walk, base, s)];
                acc = x > acc ? x : acc;
            }
            partial[t] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            uint n = min(threads, chunks - first);
            for (uint k = 0; k < n; k++) total = partial[k] > total ? partial[k] : total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0) dst[row] = total;
}

// Records the first out-of-range index: errors[0] counts reports, and only the
// first reporter writes the id and the row count.
static void report(device atomic_int* errors, int id, int rows) {
    if (atomic_fetch_add_explicit(&errors[0], 1, memory_order_relaxed) == 0) {
        atomic_store_explicit(&errors[1], id, memory_order_relaxed);
        atomic_store_explicit(&errors[2], rows, memory_order_relaxed);
    }
}

// Output element i is column i % width of table row ids[i / width].
kernel void gather_f32(device const float* table [[buffer(0)]],
                       device const int* ids [[buffer(1)]],
                       device float* dst [[buffer(2)]],
                       device atomic_int* errors [[buffer(3)]],
                       constant View& vt [[buffer(4)]],
                       constant View& vi [[buffer(5)]],
                       constant uint& width [[buffer(6)]],
                       constant int& rows [[buffer(7)]],
                       constant uint& n [[buffer(8)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    int id = ids[offset_of(vi, i / width)];
    if (id < 0 || id >= rows) {
        report(errors, id, rows);
        dst[i] = 0.0f;
        return;
    }
    dst[i] = table[offset_of(vt, id * width + i % width)];
}

// Thread j owns output column j: it zeroes the column, then adds source rows into
// the rows their ids select, in index order, as the host does.
kernel void scatter_add_f32(device const float* src [[buffer(0)]],
                            device const int* ids [[buffer(1)]],
                            device float* dst [[buffer(2)]],
                            device atomic_int* errors [[buffer(3)]],
                            constant View& vs [[buffer(4)]],
                            constant View& vi [[buffer(5)]],
                            constant uint& width [[buffer(6)]],
                            constant int& rows [[buffer(7)]],
                            constant uint& n [[buffer(8)]],
                            uint j [[thread_position_in_grid]]) {
    if (j >= width) return;
    for (int r = 0; r < rows; r++) dst[r * width + j] = 0.0f;
    for (uint i = 0; i < n; i++) {
        int id = ids[offset_of(vi, i)];
        if (id < 0 || id >= rows) {
            report(errors, id, rows);
            continue;
        }
        dst[id * width + j] += src[offset_of(vs, i * width + j)];
    }
}

// How to invert a view: its dimensions with a nonzero stride and more than one
// index, by decreasing stride, then the rest (stride 0 or one index), in order.
struct Inverse {
    int nstrided;
    int strided[MAX_DIMS];
    int nrepeated;
    int repeated[MAX_DIMS];
};

// Output position p is the sum of the source elements that view vv reads from
// p, in logical order, as the host's index-order accumulation adds them.
//
// 1. Decomposes p - offset over the strided dimensions, largest stride first.
//    If nothing remains and every index is in range, the view reads p.
// 2. Walks the repeated dimensions in row-major order; each combination is one
//    logical element that reads p.
kernel void unview_f32(device const float* src [[buffer(0)]],
                       device float* dst [[buffer(1)]],
                       constant View& vs [[buffer(2)]],
                       constant View& vv [[buffer(3)]],
                       constant Inverse& inv [[buffer(4)]],
                       constant uint& n [[buffer(5)]],
                       uint p [[thread_position_in_grid]]) {
    if (p >= n) return;
    int idx[MAX_DIMS];
    for (int d = 0; d < MAX_DIMS; d++) idx[d] = 0;
    int rem = int(p) - vv.offset;
    bool hit = rem >= 0;
    for (int k = 0; k < inv.nstrided && hit; k++) {
        int d = inv.strided[k];
        int at = rem / vv.strides[d];
        rem -= at * vv.strides[d];
        hit = at < vv.shape[d];
        idx[d] = at;
    }
    float acc = 0.0f;
    if (hit && rem == 0) {
        int total = 1;
        for (int k = 0; k < inv.nrepeated; k++) total *= vv.shape[inv.repeated[k]];
        for (int f = 0; f < total; f++) {
            int t = f;
            for (int k = inv.nrepeated - 1; k >= 0; k--) {
                int d = inv.repeated[k];
                idx[d] = t % vv.shape[d];
                t /= vv.shape[d];
            }
            int flat = 0;
            for (int d = 0; d < vv.ndim; d++) flat = flat * vv.shape[d] + idx[d];
            acc += src[offset_of(vs, flat)];
        }
    }
    dst[p] = acc;
}

// Output element i of a batched matrix product, [..., m, k] by [..., k, n], as a
// dot product over k in index order, as the host does. dims holds m, k and n.
//
// Each batch matrix's start comes from offset_of once; rows, columns and k then
// step by the views' last two strides. The product and the sum are separate
// statements, so the compiler cannot fuse them into one rounding.
kernel void matmul_f32(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float* dst [[buffer(2)]],
                       constant View& va [[buffer(3)]],
                       constant View& vb [[buffer(4)]],
                       constant uint* dims [[buffer(5)]],
                       constant uint& total [[buffer(6)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= total) return;
    uint m = dims[0], k = dims[1], n = dims[2];
    uint col = i % n;
    uint row = (i / n) % m;
    uint batch = i / (m * n);
    int pa = offset_of(va, batch * m * k) + int(row) * va.strides[va.ndim - 2];
    int pb = offset_of(vb, batch * k * n) + int(col) * vb.strides[vb.ndim - 1];
    int sa = va.strides[va.ndim - 1];
    int sb = vb.strides[vb.ndim - 2];
    float acc = 0.0f;
    for (uint s = 0; s < k; s++) {
        float product = a[pa + int(s) * sa] * b[pb + int(s) * sb];
        acc = acc + product;
    }
    dst[i] = acc;
}

// Split-K matrix products, for when one thread per output leaves the GPU idle, as
// in a matrix-vector product.
//
// A threadgroup owns COLS adjacent output columns of one output row, with SLICES
// threads per column. k is cut into chunks of CHUNK elements at fixed offsets, so
// a chunk boundary never depends on k, m or n. Each thread sums whole chunks in
// index order, and adjacent threads read adjacent columns of b, so reads coalesce.
#define COLS 32
#define SLICES 8
#define CHUNK 64

// Returns the sum of chunk c of the dot product for one output, in index order.
static float chunk_sum(device const float* a, device const float* b, int pa, int pb,
                       int sa, int sb, uint k, uint c) {
    float acc = 0.0f;
    uint stop = min(k, (c + 1) * CHUNK);
    for (uint s = c * CHUNK; s < stop; s++) {
        float product = a[pa + int(s) * sa] * b[pb + int(s) * sb];
        acc = acc + product;
    }
    return acc;
}

// Deterministic: in each pass the SLICES threads of a column compute SLICES
// consecutive chunks, and the first thread adds them to the column's total in
// chunk order. The total is therefore ((c0 + c1) + c2) + ..., whatever the
// threadgroup size; extra chunks past k add exact zeros.
kernel void matmul_split_f32(device const float* a [[buffer(0)]],
                             device const float* b [[buffer(1)]],
                             device float* dst [[buffer(2)]],
                             constant View& va [[buffer(3)]],
                             constant View& vb [[buffer(4)]],
                             constant uint* dims [[buffer(5)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint2 tid [[thread_position_in_threadgroup]]) {
    threadgroup float partial[SLICES][COLS];
    uint m = dims[0], k = dims[1], n = dims[2];
    uint lane = tid.x, slice = tid.y;
    uint col = group.x * COLS + lane;
    uint out_row = group.y;
    uint batch = out_row / m, row = out_row % m;
    bool live = col < n;
    int pa = offset_of(va, batch * m * k) + int(row) * va.strides[va.ndim - 2];
    int pb = live ? offset_of(vb, batch * k * n) + int(col) * vb.strides[vb.ndim - 1] : 0;
    int sa = va.strides[va.ndim - 1];
    int sb = vb.strides[vb.ndim - 2];
    uint chunks = (k + CHUNK - 1) / CHUNK;
    float total = 0.0f;
    for (uint first = 0; first < chunks; first += SLICES) {
        uint c = first + slice;
        partial[slice][lane] = live && c < chunks ? chunk_sum(a, b, pa, pb, sa, sb, k, c) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (slice == 0) {
            for (uint t = 0; t < SLICES; t++) total = total + partial[t][lane];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (slice == 0 && live) dst[out_row * n + col] = total;
}

// Tiled matrix products, for when there are enough rows to share operands.
//
// A threadgroup owns a BM x BN block of outputs of one batch matrix, and each of
// its (BM / TM) x (BN / TN) threads a TM x TN patch, whose rows and columns are
// strided by the thread grid so that neighbouring threads read neighbouring
// memory. k is walked in tiles of BK, which divides CHUNK: the threads first copy
// a BM x BK tile of a and a BK x BN tile of b into threadgroup memory, each
// element once, then every thread reads its operands from there.
//
// Bit-identical to matmul_split_f32, so the two can be chosen by speed alone:
// each output sums every chunk of CHUNK in index order from zero, and adds the
// chunk sums to its total in chunk order, never reading past k; it then adds the
// exact zero split-K adds for chunks past k when their number is not a multiple
// of SLICES, which turns a total of -0 into +0 as split-K does.
template <uint BM, uint BN, uint BK, uint TM, uint TN>
kernel void matmul_tiled(device const float* a [[buffer(0)]],
                         device const float* b [[buffer(1)]],
                         device float* dst [[buffer(2)]],
                         constant View& va [[buffer(3)]],
                         constant View& vb [[buffer(4)]],
                         constant uint* dims [[buffer(5)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint3 tid [[thread_position_in_threadgroup]]) {
    constexpr uint TX = BN / TN, TY = BM / TM, THREADS = TX * TY;
    // a's tile is stored k-major and b's row-major, each row padded by one, so
    // that the reads of the inner loop and most stores avoid bank conflicts.
    threadgroup float as[BK * (BM + 1)];
    threadgroup float bs[BK * (BN + 1)];
    uint m = dims[0], k = dims[1], n = dims[2];
    uint batch = group.z, row0 = group.y * BM, col0 = group.x * BN;
    uint t = tid.y * TX + tid.x;
    int pa = offset_of(va, batch * m * k);
    int pb = offset_of(vb, batch * k * n);
    int ra = va.strides[va.ndim - 2], sa = va.strides[va.ndim - 1];
    int sb = vb.strides[vb.ndim - 2], cb = vb.strides[vb.ndim - 1];
    float total[TM][TN];
    float partial[TM][TN];
    for (uint i = 0; i < TM; i++) {
        for (uint j = 0; j < TN; j++) {
            total[i][j] = 0.0f;
            partial[i][j] = 0.0f;
        }
    }
    for (uint k0 = 0; k0 < k; k0 += BK) {
        uint width = min(BK, k - k0);
        // Copies the tiles, walking whichever index is contiguous in memory
        // fastest across threads, and zero outside the matrices.
        for (uint e = t; e < BM * BK; e += THREADS) {
            uint r = sa == 1 ? e / BK : e % BM;
            uint s = sa == 1 ? e % BK : e / BM;
            bool inside = row0 + r < m && s < width;
            as[s * (BM + 1) + r] = inside ? a[pa + int(row0 + r) * ra + int(k0 + s) * sa] : 0.0f;
        }
        for (uint e = t; e < BK * BN; e += THREADS) {
            uint s = cb == 1 ? e / BN : e % BK;
            uint c = cb == 1 ? e % BN : e / BK;
            bool inside = col0 + c < n && s < width;
            bs[s * (BN + 1) + c] = inside ? b[pb + int(k0 + s) * sb + int(col0 + c) * cb] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = 0; s < width; s++) {
            float x[TM], y[TN];
            for (uint i = 0; i < TM; i++) x[i] = as[s * (BM + 1) + tid.y + i * TY];
            for (uint j = 0; j < TN; j++) y[j] = bs[s * (BN + 1) + tid.x + j * TX];
            for (uint i = 0; i < TM; i++) {
                for (uint j = 0; j < TN; j++) {
                    float product = x[i] * y[j];
                    partial[i][j] = partial[i][j] + product;
                }
            }
        }
        if ((k0 + BK) % CHUNK == 0 || k0 + BK >= k) {
            for (uint i = 0; i < TM; i++) {
                for (uint j = 0; j < TN; j++) {
                    total[i][j] = total[i][j] + partial[i][j];
                    partial[i][j] = 0.0f;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint chunks = (k + CHUNK - 1) / CHUNK;
    bool padded = chunks % SLICES != 0;
    for (uint i = 0; i < TM; i++) {
        uint row = row0 + tid.y + i * TY;
        for (uint j = 0; j < TN; j++) {
            uint col = col0 + tid.x + j * TX;
            if (row < m && col < n) dst[(batch * m + row) * n + col] = padded ? total[i][j] + 0.0f : total[i][j];
        }
    }
}

typedef decltype(matmul_tiled<32, 32, 64, 4, 4>) matmul_tiled_32_t;
typedef decltype(matmul_tiled<64, 64, 32, 4, 4>) matmul_tiled_64_t;
typedef decltype(matmul_tiled<32, 64, 32, 4, 4>) matmul_tiled_32x64_t;
template [[host_name("matmul_tiled_32x32_f32")]] kernel matmul_tiled_32_t matmul_tiled<32, 32, 64, 4, 4>;
template [[host_name("matmul_tiled_64x64_f32")]] kernel matmul_tiled_64_t matmul_tiled<64, 64, 32, 4, 4>;
template [[host_name("matmul_tiled_32x64_f32")]] kernel matmul_tiled_32x64_t matmul_tiled<32, 64, 32, 4, 4>;

#if __METAL_VERSION__ >= 300
kernel void fill_zero_f32(device float* dst [[buffer(0)]],
                          constant uint& n [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < n) dst[i] = 0.0f;
}

// Atomic, for measurement only: each thread adds its chunk sums straight into the
// zeroed output, so the order of the additions, and the bits, can change from
// run to run.
kernel void matmul_atomic_f32(device const float* a [[buffer(0)]],
                              device const float* b [[buffer(1)]],
                              device atomic_float* dst [[buffer(2)]],
                              constant View& va [[buffer(3)]],
                              constant View& vb [[buffer(4)]],
                              constant uint* dims [[buffer(5)]],
                              uint2 group [[threadgroup_position_in_grid]],
                              uint2 tid [[thread_position_in_threadgroup]]) {
    uint m = dims[0], k = dims[1], n = dims[2];
    uint lane = tid.x, slice = tid.y;
    uint col = group.x * COLS + lane;
    if (col >= n) return;
    uint out_row = group.y;
    uint batch = out_row / m, row = out_row % m;
    int pa = offset_of(va, batch * m * k) + int(row) * va.strides[va.ndim - 2];
    int pb = offset_of(vb, batch * k * n) + int(col) * vb.strides[vb.ndim - 1];
    int sa = va.strides[va.ndim - 1];
    int sb = vb.strides[vb.ndim - 2];
    uint chunks = (k + CHUNK - 1) / CHUNK;
    float acc = 0.0f;
    for (uint c = slice; c < chunks; c += SLICES) acc = acc + chunk_sum(a, b, pa, pb, sa, sb, k, c);
    atomic_fetch_add_explicit(&dst[out_row * n + col], acc, memory_order_relaxed);
}
#endif

// Fused row ops: one threadgroup per row along the last axis. Each pass folds the
// row's chunks of REDUCE_CHUNK at fixed offsets, a thread per chunk, and the
// first thread folds the chunk results in chunk order, as the reductions above
// do, so a row padded with -inf (softmax) keeps its bits. Each thread then
// rewrites only the chunks it wrote itself. A row lies along the last dimension,
// so its elements are found from its start by the last stride.
//
// Returns the maximum of row `base` of `n` elements of `a` through `v`.
static float row_max(device const float* a, constant View& v, uint base, uint n,
                     threadgroup float* partial, threadgroup float* shared,
                     uint t, uint threads) {
    uint chunks = (n + REDUCE_CHUNK - 1) / REDUCE_CHUNK;
    int start = offset_of(v, base);
    int stride = run_stride(v);
    float top = a[start];
    for (uint first = 0; first < chunks; first += threads) {
        uint c = first + t;
        if (c < chunks) {
            uint stop = min(n, (c + 1) * REDUCE_CHUNK);
            float acc = a[start + int(c * REDUCE_CHUNK) * stride];
            for (uint s = c * REDUCE_CHUNK + 1; s < stop; s++) {
                float x = a[start + int(s) * stride];
                acc = x > acc ? x : acc;
            }
            partial[t] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            uint m = min(threads, chunks - first);
            for (uint k = 0; k < m; k++) top = partial[k] > top ? partial[k] : top;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0) *shared = top;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *shared;
}

// Returns the sum, in chunk order, of the partial sums each thread puts in
// `partial` for the chunks it owns, one pass of `threads` chunks at a time.
#define ROW_SUM(total, chunk_sum)                                              \
    {                                                                          \
        uint chunks = (n + REDUCE_CHUNK - 1) / REDUCE_CHUNK;                   \
        total = 0.0f;                                                          \
        for (uint first = 0; first < chunks; first += threads) {               \
            uint c = first + t;                                                \
            float acc = 0.0f;                                                  \
            if (c < chunks) {                                                  \
                uint stop = min(n, (c + 1) * REDUCE_CHUNK);                    \
                for (uint s = c * REDUCE_CHUNK; s < stop; s++) { chunk_sum }   \
            }                                                                  \
            partial[t] = acc;                                                  \
            threadgroup_barrier(mem_flags::mem_threadgroup);                   \
            if (t == 0) {                                                      \
                uint m = min(threads, chunks - first);                         \
                for (uint k = 0; k < m; k++) total = total + partial[k];       \
            }                                                                  \
            threadgroup_barrier(mem_flags::mem_threadgroup);                   \
        }                                                                      \
        if (t == 0) shared = total;                                           \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        total = shared;                                                       \
    }

// Visits every element of the chunks this thread owns.
#define OWN_ELEMENTS(body)                                                     \
    for (uint c = t; c < (n + REDUCE_CHUNK - 1) / REDUCE_CHUNK; c += threads) {\
        uint stop = min(n, (c + 1) * REDUCE_CHUNK);                            \
        for (uint s = c * REDUCE_CHUNK; s < stop; s++) { body }                \
    }

kernel void softmax_f32(device const float* a [[buffer(0)]],
                        device float* dst [[buffer(1)]],
                        constant View& v [[buffer(2)]],
                        constant uint& n [[buffer(3)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint t [[thread_position_in_threadgroup]],
                        uint threads [[threads_per_threadgroup]]) {
    threadgroup float partial[REDUCE_THREADS];
    threadgroup float shared;
    uint base = row * n;
    int start = offset_of(v, base);
    int stride = run_stride(v);
    float top = row_max(a, v, base, n, partial, &shared, t, threads);
    float total;
    ROW_SUM(total, float e = precise::exp(a[start + int(s) * stride] - top); dst[base + s] = e; acc = acc + e;)
    OWN_ELEMENTS(dst[base + s] = dst[base + s] / total;)
}

kernel void log_softmax_f32(device const float* a [[buffer(0)]],
                            device float* dst [[buffer(1)]],
                            constant View& v [[buffer(2)]],
                            constant uint& n [[buffer(3)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint t [[thread_position_in_threadgroup]],
                            uint threads [[threads_per_threadgroup]]) {
    threadgroup float partial[REDUCE_THREADS];
    threadgroup float shared;
    uint base = row * n;
    int start = offset_of(v, base);
    int stride = run_stride(v);
    float top = row_max(a, v, base, n, partial, &shared, t, threads);
    float total;
    ROW_SUM(total, float shifted = a[start + int(s) * stride] - top; dst[base + s] = shifted; acc = acc + precise::exp(shifted);)
    float log_total = precise::log(total);
    OWN_ELEMENTS(dst[base + s] = dst[base + s] - log_total;)
}

kernel void layernorm_f32(device const float* a [[buffer(0)]],
                          device const float* w [[buffer(1)]],
                          device const float* b [[buffer(2)]],
                          device float* dst [[buffer(3)]],
                          constant View& v [[buffer(4)]],
                          constant View& vw [[buffer(5)]],
                          constant View& vb [[buffer(6)]],
                          constant uint& n [[buffer(7)]],
                          constant float& epsilon [[buffer(8)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint t [[thread_position_in_threadgroup]],
                          uint threads [[threads_per_threadgroup]]) {
    threadgroup float partial[REDUCE_THREADS];
    threadgroup float shared;
    uint base = row * n;
    int start = offset_of(v, base);
    int stride = run_stride(v);
    int wstart = offset_of(vw, base);
    int bstart = offset_of(vb, base);
    float count = float(n);
    float total;
    ROW_SUM(total, acc = acc + a[start + int(s) * stride];)
    float mean = total / count;
    float squares;
    ROW_SUM(squares, float centred = a[start + int(s) * stride] - mean; dst[base + s] = centred; float sq = centred * centred; acc = acc + sq;)
    float variance = squares / count;
    float deviation = precise::sqrt(variance + epsilon);
    OWN_ELEMENTS(float scaled = dst[base + s] / deviation; float weighted = scaled * w[wstart + int(s) * run_stride(vw)]; dst[base + s] = weighted + b[bstart + int(s) * run_stride(vb)];)
}
