// Hand-written kernels for the CUDA device, compiled by NVRTC after prelude.cu
// with fused multiply-adds off, so +, -, * and comparisons give the same bits as
// the host kernels. Each is a port of the Metal kernel of the same name.

#define BINARY(name, expr)                                                          \
    extern "C" __global__ void name(const float* a, const float* b, float* dst,    \
                                    View va, View vb, unsigned int n) {             \
        unsigned int i = thread_index();                                            \
        if (i >= n) return;                                                         \
        float x = a[offset_of(va, i)];                                              \
        float y = b[offset_of(vb, i)];                                              \
        dst[i] = (expr);                                                            \
    }

#define UNARY(name, expr)                                                           \
    extern "C" __global__ void name(const float* a, float* dst, View va, unsigned int n) { \
        unsigned int i = thread_index();                                            \
        if (i >= n) return;                                                         \
        float x = a[offset_of(va, i)];                                              \
        dst[i] = (expr);                                                            \
    }

BINARY(add_f32, x + y)
BINARY(sub_f32, x - y)
BINARY(mul_f32, x * y)
BINARY(div_f32, x / y)
BINARY(maximum_f32, x > y ? x : y)
BINARY(greater_equal_f32, x >= y ? 1.0f : 0.0f)

UNARY(exp_f32, expf(x))
UNARY(log_f32, logf(x))
UNARY(tanh_f32, tanhf(x))
UNARY(sqrt_f32, sqrtf(x))
UNARY(copy_f32, x)

// Copies source element i into output position offset_of(vd, i): Copy into a
// region of the output, as Cat does for each source.
extern "C" __global__ void copy_into_f32(const float* src, float* dst, View vs, View vd, unsigned int n) {
    unsigned int i = thread_index();
    if (i >= n) return;
    dst[offset_of(vd, i)] = src[offset_of(vs, i)];
}

// Reductions fold each output's run of `run` consecutive logical indices of `v`,
// in chunks of REDUCE_CHUNK at fixed offsets, as the Metal kernels do: each
// thread of an output's block folds whole chunks in index order, and the first
// thread folds the chunk results in chunk order. A run of one chunk or less is
// folded by one thread per output, the same arithmetic as a single chunk.
#define REDUCE_THREADS 256
#define REDUCE_CHUNK 16

// Returns the position of element s of the run that starts at logical index base.
__device__ int run_position(const View& v, int start, unsigned int walk, unsigned int base, unsigned int s) {
    return walk ? start + int(s) * run_stride(v) : offset_of(v, base + s);
}

extern "C" __global__ void sum_short_f32(const float* a, float* dst, View v, unsigned int run,
                                         unsigned int count, unsigned int walk) {
    unsigned int row = thread_index();
    if (row >= count) return;
    int start = offset_of(v, row * run);
    float acc = 0.0f;
    for (unsigned int j = 0; j < run; j++) acc = acc + a[run_position(v, start, walk, row * run, j)];
    dst[row] = 0.0f + acc;
}

extern "C" __global__ void max_short_f32(const float* a, float* dst, View v, unsigned int run,
                                         unsigned int count, unsigned int walk) {
    unsigned int row = thread_index();
    if (row >= count) return;
    int start = offset_of(v, row * run);
    float acc = a[start];
    for (unsigned int j = 1; j < run; j++) {
        float x = a[run_position(v, start, walk, row * run, j)];
        acc = x > acc ? x : acc;
    }
    dst[row] = acc;
}

extern "C" __global__ void sum_f32(const float* a, float* dst, View v, unsigned int run, unsigned int walk) {
    __shared__ float partial[REDUCE_THREADS];
    unsigned int row = blockIdx.x, t = threadIdx.x, threads = blockDim.x;
    unsigned int base = row * run;
    int start = offset_of(v, base);
    unsigned int chunks = (run + REDUCE_CHUNK - 1) / REDUCE_CHUNK;
    float total = 0.0f;
    for (unsigned int first = 0; first < chunks; first += threads) {
        unsigned int c = first + t;
        float acc = 0.0f;
        if (c < chunks) {
            unsigned int stop = min(run, (c + 1) * REDUCE_CHUNK);
            for (unsigned int s = c * REDUCE_CHUNK; s < stop; s++) acc = acc + a[run_position(v, start, walk, base, s)];
        }
        partial[t] = acc;
        __syncthreads();
        if (t == 0) {
            unsigned int n = min(threads, chunks - first);
            for (unsigned int k = 0; k < n; k++) total = total + partial[k];
        }
        __syncthreads();
    }
    if (t == 0) dst[row] = total;
}

extern "C" __global__ void max_f32(const float* a, float* dst, View v, unsigned int run, unsigned int walk) {
    __shared__ float partial[REDUCE_THREADS];
    unsigned int row = blockIdx.x, t = threadIdx.x, threads = blockDim.x;
    unsigned int base = row * run;
    int start = offset_of(v, base);
    unsigned int chunks = (run + REDUCE_CHUNK - 1) / REDUCE_CHUNK;
    float total = a[start];
    for (unsigned int first = 0; first < chunks; first += threads) {
        unsigned int c = first + t;
        if (c < chunks) {
            unsigned int stop = min(run, (c + 1) * REDUCE_CHUNK);
            float acc = a[run_position(v, start, walk, base, c * REDUCE_CHUNK)];
            for (unsigned int s = c * REDUCE_CHUNK + 1; s < stop; s++) {
                float x = a[run_position(v, start, walk, base, s)];
                acc = x > acc ? x : acc;
            }
            partial[t] = acc;
        }
        __syncthreads();
        if (t == 0) {
            unsigned int n = min(threads, chunks - first);
            for (unsigned int k = 0; k < n; k++) total = partial[k] > total ? partial[k] : total;
        }
        __syncthreads();
    }
    if (t == 0) dst[row] = total;
}

// Records the first out-of-range index: errors[0] counts reports, and only the
// first reporter writes the id and the row count.
__device__ void report(int* errors, int id, int rows) {
    if (atomicAdd(&errors[0], 1) == 0) {
        errors[1] = id;
        errors[2] = rows;
    }
}

// Output element i is column i % width of table row ids[i / width].
extern "C" __global__ void gather_f32(const float* table, const int* ids, float* dst, int* errors,
                                      View vt, View vi, unsigned int width, int rows, unsigned int n) {
    unsigned int i = thread_index();
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
extern "C" __global__ void scatter_add_f32(const float* src, const int* ids, float* dst, int* errors,
                                           View vs, View vi, unsigned int width, int rows, unsigned int n) {
    unsigned int j = thread_index();
    if (j >= width) return;
    for (int r = 0; r < rows; r++) dst[r * width + j] = 0.0f;
    for (unsigned int i = 0; i < n; i++) {
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
extern "C" __global__ void unview_f32(const float* src, float* dst, View vs, View vv, Inverse inv, unsigned int n) {
    unsigned int p = thread_index();
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
// statements, and NVRTC runs with fused multiply-adds off, so each rounds.
extern "C" __global__ void matmul_f32(const float* a, const float* b, float* dst, View va, View vb,
                                      const float* c, View vc, unsigned int m, unsigned int k, unsigned int n,
                                      unsigned int total) {
    unsigned int i = thread_index();
    if (i >= total) return;
    unsigned int col = i % n;
    unsigned int row = (i / n) % m;
    unsigned int batch = i / (m * n);
    int pa = offset_of(va, batch * m * k) + int(row) * va.strides[va.ndim - 2];
    int pb = offset_of(vb, batch * k * n) + int(col) * vb.strides[vb.ndim - 1];
    int sa = va.strides[va.ndim - 1];
    int sb = vb.strides[vb.ndim - 2];
    float acc = 0.0f;
    for (unsigned int s = 0; s < k; s++) {
        float product = a[pa + int(s) * sa] * b[pb + int(s) * sb];
        acc = acc + product;
    }
    dst[i] = with_bias(acc, c, vc, i);
}

// Split-K matrix products, for when one thread per output leaves the GPU idle, as
// in a matrix-vector product.
//
// A block owns COLS adjacent output columns of one output row, with SLICES
// threads per column. k is cut into chunks of CHUNK elements at fixed offsets, so
// a chunk boundary never depends on k, m or n. Each thread sums whole chunks in
// index order, and adjacent threads read adjacent columns of b.
#define COLS 32
#define SLICES 8
#define CHUNK 64

// Returns the sum of chunk c of the dot product for one output, in index order.
__device__ float chunk_sum(const float* a, const float* b, int pa, int pb, int sa, int sb,
                           unsigned int k, unsigned int c) {
    float acc = 0.0f;
    unsigned int stop = min(k, (c + 1) * CHUNK);
    for (unsigned int s = c * CHUNK; s < stop; s++) {
        float product = a[pa + int(s) * sa] * b[pb + int(s) * sb];
        acc = acc + product;
    }
    return acc;
}

// Deterministic: in each pass the SLICES threads of a column compute SLICES
// consecutive chunks, and the first thread adds them to the column's total in
// chunk order. The total is therefore ((c0 + c1) + c2) + ..., whatever the
// block size; extra chunks past k add exact zeros.
extern "C" __global__ void matmul_split_f32(const float* a, const float* b, float* dst, View va, View vb,
                                            const float* c, View vc, unsigned int m, unsigned int k,
                                            unsigned int n) {
    __shared__ float partial[SLICES][COLS];
    unsigned int lane = threadIdx.x, slice = threadIdx.y;
    unsigned int col = blockIdx.x * COLS + lane;
    unsigned int out_row = blockIdx.y;
    unsigned int batch = out_row / m, row = out_row % m;
    bool live = col < n;
    int pa = offset_of(va, batch * m * k) + int(row) * va.strides[va.ndim - 2];
    int pb = live ? offset_of(vb, batch * k * n) + int(col) * vb.strides[vb.ndim - 1] : 0;
    int sa = va.strides[va.ndim - 1];
    int sb = vb.strides[vb.ndim - 2];
    unsigned int chunks = (k + CHUNK - 1) / CHUNK;
    float total = 0.0f;
    for (unsigned int first = 0; first < chunks; first += SLICES) {
        unsigned int c = first + slice;
        partial[slice][lane] = live && c < chunks ? chunk_sum(a, b, pa, pb, sa, sb, k, c) : 0.0f;
        __syncthreads();
        if (slice == 0) {
            for (unsigned int t = 0; t < SLICES; t++) total = total + partial[t][lane];
        }
        __syncthreads();
    }
    if (slice == 0 && live) dst[out_row * n + col] = with_bias(total, c, vc, out_row * n + col);
}

// Two-pass matrix products, for products of few rows, as in decoding one token,
// where split-K has too few blocks and reads a transposed b across its rows.
//
// The first pass launches a block per DECODE_COLS adjacent output columns, chunk
// of CHUNK elements of k, and DECODE_ROWS rows of each batch matrix. The block
// copies its chunk of b into shared memory along whichever axis is contiguous,
// and its rows' chunk of a, then each thread sums its column's chunk in index
// order for each row, writing the sums to scratch, [batch][chunk][row][column].
// The second pass adds each output's chunk sums in chunk order from zero.
//
// Bit-identical to matmul_split_f32: the same chunk sums, added in the same
// order, with the exact zero split-K adds when the chunks are not a multiple of
// SLICES (see matmul_tiled).
#define DECODE_COLS 128u
#define DECODE_ROWS 16u

extern "C" __global__ void matmul_chunks_f32(const float* a, const float* b, float* scratch, View va, View vb,
                                             unsigned int m, unsigned int k, unsigned int n) {
    __shared__ float as[DECODE_ROWS * CHUNK];
    __shared__ float bs[CHUNK * (DECODE_COLS + 1)];
    unsigned int t = threadIdx.x, col0 = blockIdx.x * DECODE_COLS, c = blockIdx.y;
    unsigned int groups = (m + DECODE_ROWS - 1) / DECODE_ROWS;
    unsigned int batch = blockIdx.z / groups, row0 = blockIdx.z % groups * DECODE_ROWS;
    unsigned int rows = min(DECODE_ROWS, m - row0);
    unsigned int k0 = c * CHUNK, width = min(k - k0, unsigned(CHUNK));
    int pa = offset_of(va, batch * m * k);
    int pb = offset_of(vb, batch * k * n);
    int ra = va.strides[va.ndim - 2], sa = va.strides[va.ndim - 1];
    int sb = vb.strides[vb.ndim - 2], cb = vb.strides[vb.ndim - 1];
    for (unsigned int e = t; e < rows * CHUNK; e += DECODE_COLS) {
        unsigned int r = e / CHUNK, s = e % CHUNK;
        as[e] = s < width ? a[pa + int(row0 + r) * ra + int(k0 + s) * sa] : 0.0f;
    }
    for (unsigned int e = t; e < CHUNK * DECODE_COLS; e += DECODE_COLS) {
        unsigned int s = cb == 1 ? e / DECODE_COLS : e % CHUNK;
        unsigned int col = cb == 1 ? e % DECODE_COLS : e / CHUNK;
        bool inside = col0 + col < n && s < width;
        bs[s * (DECODE_COLS + 1) + col] = inside ? b[pb + int(k0 + s) * sb + int(col0 + col) * cb] : 0.0f;
    }
    __syncthreads();
    unsigned int col = col0 + t;
    if (col >= n) return;
    unsigned int chunks = (k + CHUNK - 1) / CHUNK;
    for (unsigned int r = 0; r < rows; r++) {
        float acc = 0.0f;
        for (unsigned int s = 0; s < width; s++) {
            float product = as[r * CHUNK + s] * bs[s * (DECODE_COLS + 1) + t];
            acc = acc + product;
        }
        scratch[((batch * chunks + c) * m + row0 + r) * n + col] = acc;
    }
}

extern "C" __global__ void matmul_chunk_sums_f32(const float* scratch, float* dst, const float* c, View vc,
                                                 unsigned int m, unsigned int k, unsigned int n, unsigned int total) {
    unsigned int i = thread_index();
    if (i >= total) return;
    unsigned int col = i % n, row = (i / n) % m, batch = i / (m * n);
    unsigned int chunks = (k + CHUNK - 1) / CHUNK;
    float sum = 0.0f;
    for (unsigned int c = 0; c < chunks; c++) sum = sum + scratch[((batch * chunks + c) * m + row) * n + col];
    float out = chunks % SLICES != 0 ? sum + 0.0f : sum;
    dst[i] = with_bias(out, c, vc, i);
}

// Tiled matrix products, for when there are enough rows to share operands.
//
// A block owns a BM x BN block of outputs of one batch matrix, and each of its
// (BM / TM) x (BN / TN) threads a TM x TN patch, whose rows and columns are
// strided by the thread grid so that neighbouring threads read neighbouring
// memory. k is walked in tiles of BK, which divides CHUNK: the threads first copy
// a BM x BK tile of a and a BK x BN tile of b into shared memory, each element
// once, then every thread reads its operands from there.
//
// Bit-identical to matmul_split_f32, so the two can be chosen by speed alone:
// each output sums every chunk of CHUNK in index order from zero, and adds the
// chunk sums to its total in chunk order, never reading past k; it then adds the
// exact zero split-K adds for chunks past k when their number is not a multiple
// of SLICES, which turns a total of -0 into +0 as split-K does.
template <unsigned int BM, unsigned int BN, unsigned int BK, unsigned int TM, unsigned int TN>
__device__ void matmul_tiled(const float* a, const float* b, float* dst, const View& va, const View& vb,
                             const float* c, const View& vc, unsigned int m, unsigned int k, unsigned int n) {
    constexpr unsigned int TX = BN / TN, TY = BM / TM, THREADS = TX * TY;
    // a's tile is stored k-major and b's row-major, each row padded by one, so
    // that the reads of the inner loop and most stores avoid bank conflicts.
    __shared__ float as[BK * (BM + 1)];
    __shared__ float bs[BK * (BN + 1)];
    unsigned int batch = blockIdx.z, row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    unsigned int t = threadIdx.y * TX + threadIdx.x;
    int pa = offset_of(va, batch * m * k);
    int pb = offset_of(vb, batch * k * n);
    int ra = va.strides[va.ndim - 2], sa = va.strides[va.ndim - 1];
    int sb = vb.strides[vb.ndim - 2], cb = vb.strides[vb.ndim - 1];
    float total[TM][TN];
    float partial[TM][TN];
    for (unsigned int i = 0; i < TM; i++) {
        for (unsigned int j = 0; j < TN; j++) {
            total[i][j] = 0.0f;
            partial[i][j] = 0.0f;
        }
    }
    for (unsigned int k0 = 0; k0 < k; k0 += BK) {
        unsigned int width = min(BK, k - k0);
        // Copies the tiles, walking whichever index is contiguous in memory
        // fastest across threads, and zero outside the matrices.
        for (unsigned int e = t; e < BM * BK; e += THREADS) {
            unsigned int r = sa == 1 ? e / BK : e % BM;
            unsigned int s = sa == 1 ? e % BK : e / BM;
            bool inside = row0 + r < m && s < width;
            as[s * (BM + 1) + r] = inside ? a[pa + int(row0 + r) * ra + int(k0 + s) * sa] : 0.0f;
        }
        for (unsigned int e = t; e < BK * BN; e += THREADS) {
            unsigned int s = cb == 1 ? e / BN : e % BK;
            unsigned int c = cb == 1 ? e % BN : e / BK;
            bool inside = col0 + c < n && s < width;
            bs[s * (BN + 1) + c] = inside ? b[pb + int(k0 + s) * sb + int(col0 + c) * cb] : 0.0f;
        }
        __syncthreads();
        for (unsigned int s = 0; s < width; s++) {
            float x[TM], y[TN];
            for (unsigned int i = 0; i < TM; i++) x[i] = as[s * (BM + 1) + threadIdx.y + i * TY];
            for (unsigned int j = 0; j < TN; j++) y[j] = bs[s * (BN + 1) + threadIdx.x + j * TX];
            for (unsigned int i = 0; i < TM; i++) {
                for (unsigned int j = 0; j < TN; j++) {
                    float product = x[i] * y[j];
                    partial[i][j] = partial[i][j] + product;
                }
            }
        }
        if ((k0 + BK) % CHUNK == 0 || k0 + BK >= k) {
            for (unsigned int i = 0; i < TM; i++) {
                for (unsigned int j = 0; j < TN; j++) {
                    total[i][j] = total[i][j] + partial[i][j];
                    partial[i][j] = 0.0f;
                }
            }
        }
        __syncthreads();
    }
    unsigned int chunks = (k + CHUNK - 1) / CHUNK;
    bool padded = chunks % SLICES != 0;
    for (unsigned int i = 0; i < TM; i++) {
        unsigned int row = row0 + threadIdx.y + i * TY;
        for (unsigned int j = 0; j < TN; j++) {
            unsigned int col = col0 + threadIdx.x + j * TX;
            if (row < m && col < n) {
                unsigned int out = (batch * m + row) * n + col;
                float value = padded ? total[i][j] + 0.0f : total[i][j];
                dst[out] = with_bias(value, c, vc, out);
            }
        }
    }
}

#define TILED(name, BM, BN, BK)                                                         \
    extern "C" __global__ void name(const float* a, const float* b, float* dst, View va, View vb, \
                                    const float* c, View vc, unsigned int m, unsigned int k,     \
                                    unsigned int n) {                                            \
        matmul_tiled<BM, BN, BK, 4, 4>(a, b, dst, va, vb, c, vc, m, k, n);                     \
    }

TILED(matmul_tiled_32x32_f32, 32, 32, 64)
TILED(matmul_tiled_64x64_f32, 64, 64, 32)
TILED(matmul_tiled_32x64_f32, 32, 64, 32)

// Fused row ops: one block per row along the last axis. Each pass folds the
// row's chunks of REDUCE_CHUNK at fixed offsets, a thread per chunk, and the
// first thread folds the chunk results in chunk order, as the reductions above
// do, so a row padded with -inf (softmax) keeps its bits. Each thread then
// rewrites only the chunks it wrote itself. A row lies along the last dimension,
// so its elements are found from its start by the last stride.
//
// Returns the maximum of row `base` of `n` elements of `a` through `v`.
__device__ float row_max(const float* a, const View& v, unsigned int base, unsigned int n,
                         float* partial, float* shared, unsigned int t, unsigned int threads) {
    unsigned int chunks = (n + REDUCE_CHUNK - 1) / REDUCE_CHUNK;
    int start = offset_of(v, base);
    int stride = run_stride(v);
    float top = a[start];
    for (unsigned int first = 0; first < chunks; first += threads) {
        unsigned int c = first + t;
        if (c < chunks) {
            unsigned int stop = min(n, (c + 1) * REDUCE_CHUNK);
            float acc = a[start + int(c * REDUCE_CHUNK) * stride];
            for (unsigned int s = c * REDUCE_CHUNK + 1; s < stop; s++) {
                float x = a[start + int(s) * stride];
                acc = x > acc ? x : acc;
            }
            partial[t] = acc;
        }
        __syncthreads();
        if (t == 0) {
            unsigned int m = min(threads, chunks - first);
            for (unsigned int k = 0; k < m; k++) top = partial[k] > top ? partial[k] : top;
        }
        __syncthreads();
    }
    if (t == 0) *shared = top;
    __syncthreads();
    return *shared;
}

// Returns the sum, in chunk order, of the partial sums each thread puts in
// `partial` for the chunks it owns, one pass of `threads` chunks at a time.
#define ROW_SUM(total, chunk_sum)                                              \
    {                                                                          \
        unsigned int chunks = (n + REDUCE_CHUNK - 1) / REDUCE_CHUNK;           \
        total = 0.0f;                                                          \
        for (unsigned int first = 0; first < chunks; first += threads) {       \
            unsigned int c = first + t;                                        \
            float acc = 0.0f;                                                  \
            if (c < chunks) {                                                  \
                unsigned int stop = min(n, (c + 1) * REDUCE_CHUNK);            \
                for (unsigned int s = c * REDUCE_CHUNK; s < stop; s++) { chunk_sum } \
            }                                                                  \
            partial[t] = acc;                                                  \
            __syncthreads();                                                   \
            if (t == 0) {                                                      \
                unsigned int m = min(threads, chunks - first);                 \
                for (unsigned int k = 0; k < m; k++) total = total + partial[k]; \
            }                                                                  \
            __syncthreads();                                                   \
        }                                                                      \
        if (t == 0) shared = total;                                            \
        __syncthreads();                                                       \
        total = shared;                                                        \
    }

// Visits every element of the chunks this thread owns.
#define OWN_ELEMENTS(body)                                                     \
    for (unsigned int c = t; c < (n + REDUCE_CHUNK - 1) / REDUCE_CHUNK; c += threads) { \
        unsigned int stop = min(n, (c + 1) * REDUCE_CHUNK);                    \
        for (unsigned int s = c * REDUCE_CHUNK; s < stop; s++) { body }        \
    }

extern "C" __global__ void softmax_f32(const float* a, float* dst, View v, unsigned int n) {
    __shared__ float partial[REDUCE_THREADS];
    __shared__ float shared;
    unsigned int row = blockIdx.x, t = threadIdx.x, threads = blockDim.x;
    unsigned int base = row * n;
    int start = offset_of(v, base);
    int stride = run_stride(v);
    float top = row_max(a, v, base, n, partial, &shared, t, threads);
    float total;
    ROW_SUM(total, float e = expf(a[start + int(s) * stride] - top); dst[base + s] = e; acc = acc + e;)
    OWN_ELEMENTS(dst[base + s] = dst[base + s] / total;)
}

extern "C" __global__ void log_softmax_f32(const float* a, float* dst, View v, unsigned int n) {
    __shared__ float partial[REDUCE_THREADS];
    __shared__ float shared;
    unsigned int row = blockIdx.x, t = threadIdx.x, threads = blockDim.x;
    unsigned int base = row * n;
    int start = offset_of(v, base);
    int stride = run_stride(v);
    float top = row_max(a, v, base, n, partial, &shared, t, threads);
    float total;
    ROW_SUM(total, float shifted = a[start + int(s) * stride] - top; dst[base + s] = shifted; acc = acc + expf(shifted);)
    float log_total = logf(total);
    OWN_ELEMENTS(dst[base + s] = dst[base + s] - log_total;)
}

extern "C" __global__ void layernorm_f32(const float* a, const float* w, const float* b, float* dst,
                                         View v, View vw, View vb, unsigned int n, float epsilon) {
    __shared__ float partial[REDUCE_THREADS];
    __shared__ float shared;
    unsigned int row = blockIdx.x, t = threadIdx.x, threads = blockDim.x;
    unsigned int base = row * n;
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
    float deviation = sqrtf(variance + epsilon);
    OWN_ELEMENTS(float scaled = dst[base + s] / deviation; float weighted = scaled * w[wstart + int(s) * run_stride(vw)]; dst[base + s] = weighted + b[bstart + int(s) * run_stride(vb)];)
}
