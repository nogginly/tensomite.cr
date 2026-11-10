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

// The hash dropout masks are drawn from: two rounds of lowbias32 in wrapping
// 32-bit arithmetic, as Tensomite.dropout_hash computes it on the host.
static inline uint lowbias32(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

// Dropout: logical element i of a is kept, times scale, when its hash under
// (seed, stream) is at least threshold, and zeroed otherwise.
kernel void dropout_f32(device const float* a [[buffer(0)]],
                        device float* dst [[buffer(1)]],
                        constant View& va [[buffer(2)]],
                        constant uint& n [[buffer(3)]],
                        constant uint* mask [[buffer(4)]],
                        constant float& scale [[buffer(5)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float x = a[offset_of(va, i)];
    bool kept = lowbias32(lowbias32(i + mask[0]) ^ mask[1]) >= mask[2];
    dst[i] = kept ? x * scale : 0.0f;
}

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
#define NORM_THREADS 512

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

// Zeroes dst, one thread per element.
kernel void fill_zero_f32(device float* dst [[buffer(0)]],
                          constant uint& n [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < n) dst[i] = 0.0f;
}

// Thread j owns output column j, zeroed by fill_zero_f32: it adds source rows
// into the rows their ids select, in index order, as the host does.
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
    for (uint i = 0; i < n; i++) {
        int id = ids[offset_of(vi, i)];
        if (id < 0 || id >= rows) {
            report(errors, id, rows);
            continue;
        }
        dst[id * width + j] += src[offset_of(vs, i * width + j)];
    }
}

// ScatterAdd into one column, one thread per id. The thread holding an id's
// first occurrence adds the source rows of every occurrence, in index order,
// from zero, as the host does; rows no id selects keep the zeros fill_zero_f32
// wrote. Of the ids out of range, only the first is reported.
kernel void scatter_add_column_f32(device const float* src [[buffer(0)]],
                                   device const int* ids [[buffer(1)]],
                                   device float* dst [[buffer(2)]],
                                   device atomic_int* errors [[buffer(3)]],
                                   constant View& vs [[buffer(4)]],
                                   constant View& vi [[buffer(5)]],
                                   constant uint& width [[buffer(6)]],
                                   constant int& rows [[buffer(7)]],
                                   constant uint& n [[buffer(8)]],
                                   uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    int id = ids[offset_of(vi, i)];
    bool bad = id < 0 || id >= rows;
    for (uint k = 0; k < i; k++) {
        int other = ids[offset_of(vi, k)];
        if (bad ? (other < 0 || other >= rows) : other == id) return;
    }
    if (bad) {
        report(errors, id, rows);
        return;
    }
    float acc = 0.0f;
    for (uint k = i; k < n; k++) {
        if (ids[offset_of(vi, k)] == id) acc = acc + src[offset_of(vs, k)];
    }
    dst[id] = acc;
}

// How to invert a view: the strides, sizes and row-major strides of its
// dimensions with a nonzero stride and more than one index, by decreasing
// stride, then the sizes and row-major strides of the rest (stride 0 or one
// index), in order. The row-major strides turn indices into a logical element.
struct Inverse {
    int nstrided;
    int stride[MAX_DIMS];
    int size[MAX_DIMS];
    int step[MAX_DIMS];
    int nrepeated;
    int repeated_size[MAX_DIMS];
    int repeated_step[MAX_DIMS];
};

#define UNVIEW_AHEAD 32

// Output position p is the sum of the source elements that view vv reads from
// p, in logical order, as the host's index-order accumulation adds them.
//
// 1. Decomposes p - offset over the strided dimensions, largest stride first,
//    into the logical element whose repeated indices are all zero. If nothing
//    remains and every index is in range, the view reads p.
// 2. Walks the repeated dimensions in row-major order, last fastest, stepping
//    the logical element by their row-major strides; each combination is one
//    element that reads p.
kernel void unview_f32(device const float* src [[buffer(0)]],
                       device float* dst [[buffer(1)]],
                       constant View& vs [[buffer(2)]],
                       constant View& vv [[buffer(3)]],
                       constant Inverse& inv [[buffer(4)]],
                       constant uint& n [[buffer(5)]],
                       uint p [[thread_position_in_grid]]) {
    if (p >= n) return;
    int rem = int(p) - vv.offset;
    bool hit = rem >= 0;
    int flat = 0;
    for (int k = 0; k < inv.nstrided && hit; k++) {
        int at = rem / inv.stride[k];
        rem -= at * inv.stride[k];
        hit = at < inv.size[k];
        flat += at * inv.step[k];
    }
    float acc = 0.0f;
    if (hit && rem == 0) {
        if (inv.nrepeated == 0) {
            acc = src[offset_of(vs, flat)];
        } else if (inv.nrepeated == 1 && vs.dense) {
            // Loads UNVIEW_AHEAD elements before adding them, in order, so that
            // the loads overlap.
            int count = inv.repeated_size[0], step = inv.repeated_step[0];
            device const float* at = src + vs.offset + flat;
            int f = 0;
            for (; f + UNVIEW_AHEAD <= count; f += UNVIEW_AHEAD) {
                float x[UNVIEW_AHEAD];
                for (int j = 0; j < UNVIEW_AHEAD; j++) x[j] = at[(f + j) * step];
                for (int j = 0; j < UNVIEW_AHEAD; j++) acc += x[j];
            }
            for (; f < count; f++) acc += at[f * step];
        } else if (inv.nrepeated == 1) {
            int count = inv.repeated_size[0], step = inv.repeated_step[0];
            for (int f = 0; f < count; f++) acc += src[offset_of(vs, flat + f * step)];
        } else {
            int at[MAX_DIMS];
            int total = 1;
            for (int k = 0; k < inv.nrepeated; k++) {
                at[k] = 0;
                total *= inv.repeated_size[k];
            }
            for (int f = 0; f < total; f++) {
                acc += src[offset_of(vs, flat)];
                for (int k = inv.nrepeated - 1; k >= 0; k--) {
                    flat += inv.repeated_step[k];
                    if (++at[k] < inv.repeated_size[k]) break;
                    flat -= at[k] * inv.repeated_step[k];
                    at[k] = 0;
                }
            }
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
                       device const float* bias [[buffer(3)]],
                       constant View& va [[buffer(4)]],
                       constant View& vb [[buffer(5)]],
                       constant uint* dims [[buffer(6)]],
                       constant uint& total [[buffer(7)]],
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
    dst[i] = with_bias(acc, bias, col, dims[3]);
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
                             device const float* bias [[buffer(3)]],
                             constant View& va [[buffer(4)]],
                             constant View& vb [[buffer(5)]],
                             constant uint* dims [[buffer(6)]],
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
    if (slice == 0 && live) dst[out_row * n + col] = with_bias(total, bias, col, dims[3]);
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
                         device const float* bias [[buffer(3)]],
                         constant View& va [[buffer(4)]],
                         constant View& vb [[buffer(5)]],
                         constant uint* dims [[buffer(6)]],
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
            if (row < m && col < n) {
                uint out = (batch * m + row) * n + col;
                float value = padded ? total[i][j] + 0.0f : total[i][j];
                dst[out] = with_bias(value, bias, col, dims[3]);
            }
        }
    }
}

// Tiled products in two passes, for when there are too few blocks of outputs to
// fill the GPU, as in a weight's gradient, [m, k]·[k, n] with long k and small m
// and n.
//
// The first pass gives each threadgroup a BM x BN block of outputs of one batch
// matrix, as matmul_tiled does, and a run of dims[3] chunks of CHUNK along k. It
// sums each chunk as matmul_tiled does and writes the chunk sums to scratch,
// [batch][chunk][row][column]. The second pass, matmul_chunk_sums_f32, adds each
// output's chunk sums in chunk order from zero.
//
// Bit-identical to matmul_tiled, and so to matmul_split_f32: the same chunk
// sums, added in the same order, with the same exact zero add at the end.
template <uint BM, uint BN, uint BK, uint TM, uint TN>
kernel void matmul_tiled_chunks(device const float* a [[buffer(0)]],
                                device const float* b [[buffer(1)]],
                                device float* scratch [[buffer(2)]],
                                constant View& va [[buffer(3)]],
                                constant View& vb [[buffer(4)]],
                                constant uint* dims [[buffer(5)]],
                                uint3 group [[threadgroup_position_in_grid]],
                                uint3 tid [[thread_position_in_threadgroup]]) {
    constexpr uint TX = BN / TN, TY = BM / TM, THREADS = TX * TY;
    threadgroup float as[BK * (BM + 1)];
    threadgroup float bs[BK * (BN + 1)];
    uint m = dims[0], k = dims[1], n = dims[2], run = dims[3];
    uint chunks = (k + CHUNK - 1) / CHUNK, runs = (chunks + run - 1) / run;
    uint batch = group.z / runs, first = group.z % runs * run;
    uint row0 = group.y * BM, col0 = group.x * BN;
    uint t = tid.y * TX + tid.x;
    int pa = offset_of(va, batch * m * k);
    int pb = offset_of(vb, batch * k * n);
    int ra = va.strides[va.ndim - 2], sa = va.strides[va.ndim - 1];
    int sb = vb.strides[vb.ndim - 2], cb = vb.strides[vb.ndim - 1];
    float partial[TM][TN];
    for (uint i = 0; i < TM; i++) {
        for (uint j = 0; j < TN; j++) partial[i][j] = 0.0f;
    }
    uint stop = min(k, (first + run) * CHUNK);
    for (uint k0 = first * CHUNK; k0 < stop; k0 += BK) {
        uint width = min(BK, k - k0);
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
            uint c = k0 / CHUNK;
            for (uint i = 0; i < TM; i++) {
                uint row = row0 + tid.y + i * TY;
                for (uint j = 0; j < TN; j++) {
                    uint col = col0 + tid.x + j * TX;
                    if (row < m && col < n) scratch[((batch * chunks + c) * m + row) * n + col] = partial[i][j];
                    partial[i][j] = 0.0f;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Output i of a product whose chunk sums matmul_tiled_chunks wrote: the sums
// added in chunk order from zero, then the exact zero split-K adds for chunks
// past k when their number is not a multiple of SLICES, then the bias. dims
// holds m, k, n and whether there is a bias.
kernel void matmul_chunk_sums_f32(device const float* scratch [[buffer(0)]],
                                  device float* dst [[buffer(1)]],
                                  device const float* bias [[buffer(2)]],
                                  constant uint* dims [[buffer(3)]],
                                  constant uint& total [[buffer(4)]],
                                  uint i [[thread_position_in_grid]]) {
    if (i >= total) return;
    uint m = dims[0], k = dims[1], n = dims[2];
    uint col = i % n, row = (i / n) % m, batch = i / (m * n);
    uint chunks = (k + CHUNK - 1) / CHUNK;
    float sum = 0.0f;
    for (uint c = 0; c < chunks; c++) sum = sum + scratch[((batch * chunks + c) * m + row) * n + col];
    float out = chunks % SLICES != 0 ? sum + 0.0f : sum;
    dst[i] = with_bias(out, bias, col, dims[3]);
}

typedef decltype(matmul_tiled_chunks<64, 64, 32, 4, 4>) matmul_tiled_chunks_64_t;
template [[host_name("matmul_tiled_chunks_64x64_f32")]] kernel matmul_tiled_chunks_64_t matmul_tiled_chunks<64, 64, 32, 4, 4>;

typedef decltype(matmul_tiled<32, 32, 64, 4, 4>) matmul_tiled_32_t;
typedef decltype(matmul_tiled<64, 64, 32, 4, 4>) matmul_tiled_64_t;
typedef decltype(matmul_tiled<32, 64, 32, 4, 4>) matmul_tiled_32x64_t;
template [[host_name("matmul_tiled_32x32_f32")]] kernel matmul_tiled_32_t matmul_tiled<32, 32, 64, 4, 4>;
template [[host_name("matmul_tiled_64x64_f32")]] kernel matmul_tiled_64_t matmul_tiled<64, 64, 32, 4, 4>;
template [[host_name("matmul_tiled_32x64_f32")]] kernel matmul_tiled_32x64_t matmul_tiled<32, 64, 32, 4, 4>;

// Block formats (Q8_0, Q4_0, Q4_1), as ggml stores them: per block of 32, an
// F16 scale d (and for Q4_1 an F16 minimum m), then the quantised values. Each
// kernel computes what the host's `Quant` computes, operation for operation.

// Returns the half at p, little-endian, as a float, exactly.
static float block_half(device const uchar* p) {
    return float(as_type<half>(ushort(ushort(p[0]) | ushort(p[1]) << 8)));
}

// The layout of a block-format tensor: its kind (0 Q8_0, 1 Q4_0, 2 Q4_1), the
// bytes of one block, and the byte offset of its first row in the buffer.
struct Blocks {
    uint kind;
    uint bytes;
    uint offset;
};

// Returns element j of a row of blocks starting at p.
static float block_value(device const uchar* p, constant Blocks& b, uint j) {
    device const uchar* block = p + (j / 32) * b.bytes;
    uint at = j % 32;
    if (b.kind == 0) return float(as_type<char>(block[2 + at])) * block_half(block);
    uint packed = block[(b.kind == 1 ? 2 : 4) + at % 16];
    uint q = at < 16 ? (packed & 0x0f) : (packed >> 4);
    if (b.kind == 1) return float(int(q) - 8) * block_half(block);
    return block_half(block) * float(q) + block_half(block + 2);
}

// Gather from a block-format table: one thread per output element, the rows
// dequantised as they are read.
kernel void gather_blocks_f32(device const uchar* table [[buffer(0)]],
                              device const int* ids [[buffer(1)]],
                              device float* dst [[buffer(2)]],
                              device atomic_int* errors [[buffer(3)]],
                              constant Blocks& b [[buffer(4)]],
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
    device const uchar* row = table + b.offset + uint(id) * (width / 32) * b.bytes;
    dst[i] = block_value(row, b, i % width);
}

// Quantises each block of 32 activations to 8 bits, as the host's
// `quantize_activations` does: one thread per block of the rows read through
// vx, writing the F16-rounded scale, d·Σq rounded to F16, and the values.
kernel void quantize_rows_q8_f32(device const float* x [[buffer(0)]],
                                 device float* scales [[buffer(1)]],
                                 device float* sums [[buffer(2)]],
                                 device char* qs [[buffer(3)]],
                                 constant View& vx [[buffer(4)]],
                                 constant uint& blocks [[buffer(5)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i >= blocks) return;
    uint base = i * 32;
    float amax = 0.0f;
    for (uint j = 0; j < 32; j++) amax = max(amax, fabs(x[offset_of(vx, base + j)]));
    float d = amax / 127.0f;
    float id = d == 0.0f ? 0.0f : 1.0f / d;
    int total = 0;
    for (uint j = 0; j < 32; j++) {
        char q = char(round(x[offset_of(vx, base + j)] * id));
        qs[base + j] = q;
        total += q;
    }
    scales[i] = float(half(d));
    sums[i] = float(half(float(total) * d));
}

// Returns the contribution of one block of a weight row, at p, to an output:
// its integer dot product with the block's quantised activations, qb, times
// both scales, plus m · d·Σq for Q4_1, with the host's expressions, so that
// adding the contributions in block order gives the host's bits.
static float block_contribution(device const uchar* p, device const char* qb, float sx, float mx, uint kind) {
    if (kind == 0) {
        int total = 0;
        for (uint j = 0; j < 32; j++) total += int(as_type<char>(p[2 + j])) * int(qb[j]);
        return float(total) * (block_half(p) * sx);
    }
    int low = 0, high = 0;
    uint base = kind == 1 ? 2 : 4;
    int offset = kind == 1 ? 8 : 0;
    for (uint j = 0; j < 16; j++) {
        uint packed = p[base + j];
        low += (int(packed & 0x0f) - offset) * int(qb[j]);
        high += (int(packed >> 4) - offset) * int(qb[j + 16]);
    }
    if (kind == 1) return float(low + high) * block_half(p) * sx;
    return (block_half(p) * sx) * float(low + high) + block_half(p + 2) * mx;
}

// x · wᵀ for a block-format weight: one thread per output, adding its blocks'
// contributions in order. For products with outputs enough to fill the GPU.
// dims is (n, k, biased).
kernel void block_matmul_f32(device const uchar* w [[buffer(0)]],
                             device const float* scales [[buffer(1)]],
                             device const float* sums [[buffer(2)]],
                             device const char* qs [[buffer(3)]],
                             device float* dst [[buffer(4)]],
                             device const float* bias [[buffer(5)]],
                             constant Blocks& b [[buffer(6)]],
                             constant uint3& dims [[buffer(7)]],
                             constant uint& count [[buffer(8)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint n = dims.x, k = dims.y, blocks = k / 32;
    uint row = i / n, col = i % n;
    device const uchar* weights = w + b.offset + col * blocks * b.bytes;
    float sum = 0.0f;
    for (uint block = 0; block < blocks; block++)
        sum += block_contribution(weights + block * b.bytes, qs + row * k + block * 32, scales[row * blocks + block],
                                  sums[row * blocks + block], b.kind);
    dst[i] = with_bias(sum, bias, col, dims.z);
}

// Block-format products on lanes: at most LANE_GROUPS SIMD groups a
// threadgroup, each computing at most LANE_OUTPUTS outputs.
#define LANE_GROUPS 4
#define LANE_OUTPUTS 4

// Returns the 4 bytes at p, which is 2-byte aligned, as one little-endian word.
static uint word_at(device const uchar* p) {
    device const ushort* h = (device const ushort*)p;
    return uint(h[0]) | uint(h[1]) << 16;
}

// Returns c plus the dot product of the four signed bytes of a with those of b.
static int dot4(uint a, uint b, int c) {
    for (uint j = 0; j < 4; j++) c += (as_type<int>(a << (24 - 8 * j)) >> 24) * (as_type<int>(b << (24 - 8 * j)) >> 24);
    return c;
}

// A block's 32 quantised activations, four to a word.
struct Words {
    uint w[8];
};

// Returns the activations of the block at qb, which is 4-byte aligned.
static Words words_at(device const char* qb) {
    device const uint* p = (device const uint*)qb;
    Words a;
    for (uint j = 0; j < 8; j++) a.w[j] = p[j];
    return a;
}

// Returns block_contribution's value for the block at p, which is 2-byte
// aligned, from its quantised activations a: the same integer dot product,
// read four bytes at a time, and the same expressions after it.
static float wide_contribution(device const uchar* p, Words a, float sx, float mx, uint kind) {
    int total = 0;
    if (kind == 0) {
        for (uint j = 0; j < 8; j++) total = dot4(word_at(p + 2 + 4 * j), a.w[j], total);
        return float(total) * (block_half(p) * sx);
    }
    device const uchar* packed = p + (kind == 1 ? 2 : 4);
    for (uint j = 0; j < 4; j++) {
        uint word = word_at(packed + 4 * j);
        total = dot4(word & 0x0f0f0f0fu, a.w[j], total);
        total = dot4((word >> 4) & 0x0f0f0f0fu, a.w[4 + j], total);
    }
    if (kind == 1) {
        int sum = 0;
        for (uint j = 0; j < 8; j++) sum = dot4(0x01010101u, a.w[j], sum);
        return float(total - 8 * sum) * block_half(p) * sx;
    }
    return (block_half(p) * sx) * float(total) + block_half(p + 2) * mx;
}

// x · wᵀ for a block-format weight with few outputs: `per` consecutive outputs
// (at most LANE_OUTPUTS) per SIMD group, `groups` a threadgroup (at most
// LANE_GROUPS). Lane l computes the contributions of blocks l, l + 32 and so
// on to each of its group's outputs, reading the weights two bytes at a time
// and the quantised activations four, once for the outputs that share a row.
// For each 32 blocks, the lanes put their contributions in threadgroup memory,
// and lane r adds those of output r in block order, as the host does. Outputs
// past the last compute the last one and write nothing. The weights must
// start at an even byte offset.
kernel void block_matmul_lanes_f32(device const uchar* w [[buffer(0)]],
                                   device const float* scales [[buffer(1)]],
                                   device const float* sums [[buffer(2)]],
                                   device const char* qs [[buffer(3)]],
                                   device float* dst [[buffer(4)]],
                                   device const float* bias [[buffer(5)]],
                                   constant Blocks& b [[buffer(6)]],
                                   constant uint3& dims [[buffer(7)]],
                                   constant uint& count [[buffer(8)]],
                                   constant uint& per [[buffer(9)]],
                                   uint group [[threadgroup_position_in_grid]],
                                   uint simd [[simdgroup_index_in_threadgroup]],
                                   uint groups [[simdgroups_per_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float parts[LANE_GROUPS * LANE_OUTPUTS * 32];
    uint n = dims.x, k = dims.y, blocks = k / 32;
    uint first = (group * groups + simd) * per;
    uint mine = simd * LANE_OUTPUTS * 32;
    uint lead = min(first, count - 1) / n;
    float sum = 0.0f;
    for (uint start = 0; start < blocks; start += 32) {
        uint block = start + lane;
        if (block < blocks) {
            Words a = words_at(qs + lead * k + block * 32);
            for (uint r = 0; r < per; r++) {
                uint i = min(first + r, count - 1);
                uint row = i / n, col = i % n;
                device const uchar* p = w + b.offset + (col * blocks + block) * b.bytes;
                uint at = row * blocks + block;
                Words own = row == lead ? a : words_at(qs + row * k + block * 32);
                parts[mine + r * 32 + lane] = wide_contribution(p, own, scales[at], sums[at], b.kind);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < per) {
            uint run = min(blocks - start, 32u);
            for (uint j = 0; j < run; j++) sum += parts[mine + lane * 32 + j];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint output = first + lane;
    if (lane < per && output < count) dst[output] = with_bias(sum, bias, output % n, dims.z);
}

#if __METAL_VERSION__ >= 300
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

// Returns element `s` of a row of `x · scale + mask`, from the row's starts and
// strides in `a` and `m`: the scale when `flags` has bit 1, the mask when it has
// bit 2, each operation its own statement, rounded as its own op would round it.
static float scaled_at(device const float* a, int start, int stride, device const float* m, int mstart, int mstride,
                       uint s, float scale, uint flags) {
    float x = a[start + int(s) * stride];
    if (flags & 1u) x = x * scale;
    if (flags & 2u) {
        float y = m[mstart + int(s) * mstride];
        x = x + y;
    }
    return x;
}

// Softmax of `x · scale + mask`, per row: the maximum, folded as row_max folds
// it; then the exponentials and their sum; then the division.
kernel void softmax_scaled_f32(device const float* a [[buffer(0)]],
                               device const float* m [[buffer(1)]],
                               device float* dst [[buffer(2)]],
                               constant View& v [[buffer(3)]],
                               constant View& vm [[buffer(4)]],
                               constant uint& n [[buffer(5)]],
                               constant float& scale [[buffer(6)]],
                               constant uint& flags [[buffer(7)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint t [[thread_position_in_threadgroup]],
                               uint threads [[threads_per_threadgroup]]) {
    threadgroup float partial[REDUCE_THREADS];
    threadgroup float shared;
    uint base = row * n;
    int start = offset_of(v, base), stride = run_stride(v);
    int mstart = offset_of(vm, base), mstride = run_stride(vm);
    uint chunks = (n + REDUCE_CHUNK - 1) / REDUCE_CHUNK;
    float top = scaled_at(a, start, stride, m, mstart, mstride, 0, scale, flags);
    for (uint first = 0; first < chunks; first += threads) {
        uint c = first + t;
        if (c < chunks) {
            uint stop = min(n, (c + 1) * REDUCE_CHUNK);
            float acc = scaled_at(a, start, stride, m, mstart, mstride, c * REDUCE_CHUNK, scale, flags);
            for (uint s = c * REDUCE_CHUNK + 1; s < stop; s++) {
                float x = scaled_at(a, start, stride, m, mstart, mstride, s, scale, flags);
                acc = x > acc ? x : acc;
            }
            partial[t] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t == 0) {
            uint k_end = min(threads, chunks - first);
            for (uint k = 0; k < k_end; k++) top = partial[k] > top ? partial[k] : top;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t == 0) shared = top;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    top = shared;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total;
    ROW_SUM(total, float e = precise::exp(scaled_at(a, start, stride, m, mstart, mstride, s, scale, flags) - top); dst[base + s] = e; acc = acc + e;)
    OWN_ELEMENTS(dst[base + s] = dst[base + s] / total;)
}

// The gradient of a softmax's input, per row: the sum of g · y, in chunk order
// as sum_f32 adds; then y · (g - sum), times the scale when `flags` has bit 1.
kernel void softmax_grad_f32(device const float* g [[buffer(0)]],
                             device const float* y [[buffer(1)]],
                             device float* dst [[buffer(2)]],
                             constant View& vg [[buffer(3)]],
                             constant View& vy [[buffer(4)]],
                             constant uint& n [[buffer(5)]],
                             constant float& scale [[buffer(6)]],
                             constant uint& flags [[buffer(7)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint t [[thread_position_in_threadgroup]],
                             uint threads [[threads_per_threadgroup]]) {
    threadgroup float partial[REDUCE_THREADS];
    threadgroup float shared;
    uint base = row * n;
    int gstart = offset_of(vg, base), gstride = run_stride(vg);
    int ystart = offset_of(vy, base), ystride = run_stride(vy);
    float total;
    ROW_SUM(total, float p = g[gstart + int(s) * gstride] * y[ystart + int(s) * ystride]; acc = acc + p;)
    OWN_ELEMENTS(
        float d = g[gstart + int(s) * gstride] - total;
        float o = y[ystart + int(s) * ystride] * d;
        if (flags & 1u) o = o * scale;
        dst[base + s] = o;)
}

// The gradient of a log-softmax's input, per row: the sum of g, in chunk order
// as sum_f32 adds; then g - exp(y) · sum. `scale` and `flags` are unused, so
// it is encoded as softmax_grad_f32 is.
kernel void log_softmax_grad_f32(device const float* g [[buffer(0)]],
                                 device const float* y [[buffer(1)]],
                                 device float* dst [[buffer(2)]],
                                 constant View& vg [[buffer(3)]],
                                 constant View& vy [[buffer(4)]],
                                 constant uint& n [[buffer(5)]],
                                 constant float& scale [[buffer(6)]],
                                 constant uint& flags [[buffer(7)]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint t [[thread_position_in_threadgroup]],
                                 uint threads [[threads_per_threadgroup]]) {
    threadgroup float partial[REDUCE_THREADS];
    threadgroup float shared;
    uint base = row * n;
    int gstart = offset_of(vg, base), gstride = run_stride(vg);
    int ystart = offset_of(vy, base), ystride = run_stride(vy);
    float total;
    ROW_SUM(total, acc = acc + g[gstart + int(s) * gstride];)
    OWN_ELEMENTS(
        float e = precise::exp(y[ystart + int(s) * ystride]);
        float p = e * total;
        dst[base + s] = g[gstart + int(s) * gstride] - p;)
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

// Layer norm's gradients, per row, with the bits of the composite of primitives:
// mean = Σx / n, centred = x - mean, σ = sqrt(Σcentred² / n + epsilon) and
// x̂ = centred / σ, each sum in chunk order as sum_f32 adds. When `with_g` is 0,
// the row of x̂; otherwise, with ĝ = g · w, ((ĝ - Σĝ / n) - x̂ · Σ(ĝ · x̂) / n) / σ.
kernel void layernorm_grad_f32(device const float* g [[buffer(0)]],
                               device const float* x [[buffer(1)]],
                               device const float* w [[buffer(2)]],
                               device float* dst [[buffer(3)]],
                               constant View& vg [[buffer(4)]],
                               constant View& vx [[buffer(5)]],
                               constant View& vw [[buffer(6)]],
                               constant uint& n [[buffer(7)]],
                               constant float& epsilon [[buffer(8)]],
                               constant uint& with_g [[buffer(9)]],
                               uint row [[threadgroup_position_in_grid]],
                               uint t [[thread_position_in_threadgroup]],
                               uint threads [[threads_per_threadgroup]]) {
    threadgroup float partial[REDUCE_THREADS];
    threadgroup float shared;
    uint base = row * n;
    int xstart = offset_of(vx, base), xstride = run_stride(vx);
    int gstart = offset_of(vg, base), gstride = run_stride(vg);
    int wstart = offset_of(vw, base), wstride = run_stride(vw);
    float count = float(n);
    float total;
    ROW_SUM(total, acc = acc + x[xstart + int(s) * xstride];)
    float mean = total / count;
    float squares;
    ROW_SUM(squares, float c = x[xstart + int(s) * xstride] - mean; float p = c * c; acc = acc + p;)
    float variance = squares / count;
    float shifted = variance + epsilon;
    float std = precise::sqrt(shifted);
    if (!with_g) {
        OWN_ELEMENTS(float c = x[xstart + int(s) * xstride] - mean; dst[base + s] = c / std;)
        return;
    }
    float scaled_total;
    ROW_SUM(scaled_total, float sc = g[gstart + int(s) * gstride] * w[wstart + int(s) * wstride]; acc = acc + sc;)
    float scaled_mean = scaled_total / count;
    float product_total;
    ROW_SUM(product_total,
            float c = x[xstart + int(s) * xstride] - mean; float xh = c / std;
            float sc = g[gstart + int(s) * gstride] * w[wstart + int(s) * wstride];
            float p = sc * xh; acc = acc + p;)
    float product_mean = product_total / count;
    OWN_ELEMENTS(
        float c = x[xstart + int(s) * xstride] - mean; float xh = c / std;
        float sc = g[gstart + int(s) * gstride] * w[wstart + int(s) * wstride];
        float t1 = sc - scaled_mean; float t2 = xh * product_mean; float t3 = t1 - t2;
        dst[base + s] = t3 / std;)
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

// Root-mean-square normalisation of each row: one threadgroup per row of a
// power of two threads, at most NORM_THREADS. Thread t adds the squares of
// elements t, t + threads and so on, so that a SIMD group reads neighbouring
// elements together; the threads' sums are then added in a tree, each element
// divided by √(Σx²/n + ε) and scaled by the weight.
kernel void rmsnorm_f32(device const float* a [[buffer(0)]],
                        device const float* w [[buffer(1)]],
                        device float* dst [[buffer(2)]],
                        constant View& v [[buffer(3)]],
                        constant View& vw [[buffer(4)]],
                        constant uint& n [[buffer(5)]],
                        constant float& epsilon [[buffer(6)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint t [[thread_position_in_threadgroup]],
                        uint threads [[threads_per_threadgroup]]) {
    threadgroup float partial[NORM_THREADS];
    uint base = row * n;
    int start = offset_of(v, base);
    int stride = run_stride(v);
    int wstart = offset_of(vw, base);
    int wstride = run_stride(vw);
    float acc = 0.0f;
    for (uint s = t; s < n; s += threads) {
        float x = a[start + int(s) * stride];
        float sq = x * x;
        acc = acc + sq;
    }
    partial[t] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = threads / 2; width > 0; width /= 2) {
        if (t < width) partial[t] = partial[t] + partial[t + width];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float root = precise::sqrt(partial[0] / float(n) + epsilon);
    for (uint s = t; s < n; s += threads) {
        float scaled = a[start + int(s) * stride] / root;
        dst[base + s] = scaled * w[wstart + int(s) * wstride];
    }
}

// Rotary embeddings: one thread per element, `x · cos + rotate_half(x) · sin`,
// where rotate_half(x) at a column of the first half is the element half a
// row on, times -1, and at one of the second half the element half a row back.
kernel void rope_f32(device const float* a [[buffer(0)]],
                     device const float* c [[buffer(1)]],
                     device const float* sn [[buffer(2)]],
                     device float* dst [[buffer(3)]],
                     constant View& va [[buffer(4)]],
                     constant View& vc [[buffer(5)]],
                     constant View& vs [[buffer(6)]],
                     constant uint& n [[buffer(7)]],
                     constant uint& count [[buffer(8)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    uint col = i % n, middle = n / 2;
    float partner = a[offset_of(va, col < middle ? i + middle : i - middle)];
    float rotated = col < middle ? partner * -1.0f : partner;
    float first = a[offset_of(va, i)] * c[offset_of(vc, i)];
    float second = rotated * sn[offset_of(vs, i)];
    dst[i] = first + second;
}
