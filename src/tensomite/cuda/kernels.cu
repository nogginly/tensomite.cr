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

// The hash dropout masks are drawn from: two rounds of lowbias32 in wrapping
// 32-bit arithmetic, as Tensomite.dropout_hash computes it on the host.
__device__ unsigned int lowbias32(unsigned int x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

// Dropout: logical element i of a is kept, times scale, when its hash under
// (seed, stream) is at least threshold, and zeroed otherwise.
extern "C" __global__ void dropout_f32(const float* a, float* dst, View va, unsigned int n,
                                       unsigned int seed, unsigned int stream, unsigned int threshold, float scale) {
    unsigned int i = thread_index();
    if (i >= n) return;
    float x = a[offset_of(va, i)];
    bool kept = lowbias32(lowbias32(i + seed) ^ stream) >= threshold;
    dst[i] = kept ? x * scale : 0.0f;
}

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
#define NORM_THREADS 512
// Block-format products on lanes: BLOCK_WARPS warps a block, each computing at
// most LANE_OUTPUTS outputs, whose contributions it keeps LANE_STRIDE floats
// apart, so that the lanes adding them read different banks of shared memory.
#define BLOCK_WARPS 4
#define LANE_OUTPUTS 32
#define LANE_STRIDE 33

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

// Zeroes dst, one thread per element.
extern "C" __global__ void fill_zero_f32(float* dst, unsigned int n) {
    unsigned int i = thread_index();
    if (i < n) dst[i] = 0.0f;
}

// Thread j owns output column j, zeroed by fill_zero_f32: it adds source rows
// into the rows their ids select, in index order, as the host does.
extern "C" __global__ void scatter_add_f32(const float* src, const int* ids, float* dst, int* errors,
                                           View vs, View vi, unsigned int width, int rows, unsigned int n) {
    unsigned int j = thread_index();
    if (j >= width) return;
    for (unsigned int i = 0; i < n; i++) {
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
extern "C" __global__ void scatter_add_column_f32(const float* src, const int* ids, float* dst, int* errors,
                                                  View vs, View vi, unsigned int width, int rows, unsigned int n) {
    unsigned int i = thread_index();
    if (i >= n) return;
    int id = ids[offset_of(vi, i)];
    bool bad = id < 0 || id >= rows;
    for (unsigned int k = 0; k < i; k++) {
        int other = ids[offset_of(vi, k)];
        if (bad ? (other < 0 || other >= rows) : other == id) return;
    }
    if (bad) {
        report(errors, id, rows);
        return;
    }
    float acc = 0.0f;
    for (unsigned int k = i; k < n; k++) {
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

// Output position p is the sum of the source elements that view vv reads from
// p, in logical order, as the host's index-order accumulation adds them.
//
// 1. Decomposes p - offset over the strided dimensions, largest stride first,
//    into the logical element whose repeated indices are all zero. If nothing
//    remains and every index is in range, the view reads p.
// 2. Walks the repeated dimensions in row-major order, last fastest, stepping
//    the logical element by their row-major strides; each combination is one
//    element that reads p.
//
// Loops run over every possible dimension, so that they unroll and index the
// arguments only by constants (see offset_of).
extern "C" __global__ void unview_f32(const float* src, float* dst, View vs, View vv, Inverse inv, unsigned int n) {
    unsigned int p = thread_index();
    if (p >= n) return;
    int rem = int(p) - vv.offset;
    bool hit = rem >= 0;
    int flat = 0;
#pragma unroll
    for (int k = 0; k < MAX_DIMS; k++) {
        if (k < inv.nstrided && hit) {
            int at = rem / inv.stride[k];
            rem -= at * inv.stride[k];
            hit = at < inv.size[k];
            flat += at * inv.step[k];
        }
    }
    float acc = 0.0f;
    if (hit && rem == 0) {
        if (inv.nrepeated == 0) {
            acc = src[offset_of(vs, flat)];
        } else if (inv.nrepeated == 1) {
            int count = inv.repeated_size[0], step = inv.repeated_step[0];
            for (int f = 0; f < count; f++) acc += src[offset_of(vs, flat + f * step)];
        } else {
            int at[MAX_DIMS];
            int total = 1;
#pragma unroll
            for (int k = 0; k < MAX_DIMS; k++) {
                at[k] = 0;
                if (k < inv.nrepeated) total *= inv.repeated_size[k];
            }
            for (int f = 0; f < total; f++) {
                acc += src[offset_of(vs, flat)];
#pragma unroll
                for (int k = MAX_DIMS - 1; k >= 0; k--) {
                    if (k < inv.nrepeated) {
                        flat += inv.repeated_step[k];
                        if (++at[k] < inv.repeated_size[k]) break;
                        flat -= at[k] * inv.repeated_step[k];
                        at[k] = 0;
                    }
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
// statements, and NVRTC runs with fused multiply-adds off, so each rounds.
extern "C" __global__ void matmul_f32(const float* a, const float* b, float* dst, View va, View vb,
                                      const float* bias, unsigned int m, unsigned int k, unsigned int n,
                                      unsigned int total) {
    unsigned int i = thread_index();
    if (i >= total) return;
    unsigned int col = i % n;
    unsigned int row = (i / n) % m;
    unsigned int batch = i / (m * n);
    int pa = offset_of(va, batch * m * k) + int(row) * stride_of(va, va.ndim - 2);
    int pb = offset_of(vb, batch * k * n) + int(col) * stride_of(vb, vb.ndim - 1);
    int sa = stride_of(va, va.ndim - 1);
    int sb = stride_of(vb, vb.ndim - 2);
    float acc = 0.0f;
    for (unsigned int s = 0; s < k; s++) {
        float product = a[pa + int(s) * sa] * b[pb + int(s) * sb];
        acc = acc + product;
    }
    dst[i] = with_bias(acc, bias, col);
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
                                            const float* bias, unsigned int m, unsigned int k,
                                            unsigned int n) {
    __shared__ float partial[SLICES][COLS];
    unsigned int lane = threadIdx.x, slice = threadIdx.y;
    unsigned int col = blockIdx.x * COLS + lane;
    unsigned int out_row = blockIdx.y;
    unsigned int batch = out_row / m, row = out_row % m;
    bool live = col < n;
    int pa = offset_of(va, batch * m * k) + int(row) * stride_of(va, va.ndim - 2);
    int pb = live ? offset_of(vb, batch * k * n) + int(col) * stride_of(vb, vb.ndim - 1) : 0;
    int sa = stride_of(va, va.ndim - 1);
    int sb = stride_of(vb, vb.ndim - 2);
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
    if (slice == 0 && live) dst[out_row * n + col] = with_bias(total, bias, col);
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
    int ra = stride_of(va, va.ndim - 2), sa = stride_of(va, va.ndim - 1);
    int sb = stride_of(vb, vb.ndim - 2), cb = stride_of(vb, vb.ndim - 1);
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

extern "C" __global__ void matmul_chunk_sums_f32(const float* scratch, float* dst, const float* bias,
                                                 unsigned int m, unsigned int k, unsigned int n, unsigned int total) {
    unsigned int i = thread_index();
    if (i >= total) return;
    unsigned int col = i % n, row = (i / n) % m, batch = i / (m * n);
    unsigned int chunks = (k + CHUNK - 1) / CHUNK;
    float sum = 0.0f;
    for (unsigned int c = 0; c < chunks; c++) sum = sum + scratch[((batch * chunks + c) * m + row) * n + col];
    float out = chunks % SLICES != 0 ? sum + 0.0f : sum;
    dst[i] = with_bias(out, bias, col);
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
                             const float* bias, unsigned int m, unsigned int k, unsigned int n) {
    constexpr unsigned int TX = BN / TN, TY = BM / TM, THREADS = TX * TY;
    // a's tile is stored k-major and b's row-major, each row padded by one, so
    // that the reads of the inner loop and most stores avoid bank conflicts.
    __shared__ float as[BK * (BM + 1)];
    __shared__ float bs[BK * (BN + 1)];
    unsigned int batch = blockIdx.z, row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    unsigned int t = threadIdx.y * TX + threadIdx.x;
    int pa = offset_of(va, batch * m * k);
    int pb = offset_of(vb, batch * k * n);
    int ra = stride_of(va, va.ndim - 2), sa = stride_of(va, va.ndim - 1);
    int sb = stride_of(vb, vb.ndim - 2), cb = stride_of(vb, vb.ndim - 1);
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
                dst[out] = with_bias(value, bias, col);
            }
        }
    }
}

#define TILED(name, BM, BN, BK)                                                         \
    extern "C" __global__ void name(const float* a, const float* b, float* dst, View va, View vb, \
                                    const float* bias, unsigned int m, unsigned int k, unsigned int n) { \
        matmul_tiled<BM, BN, BK, 4, 4>(a, b, dst, va, vb, bias, m, k, n);                  \
    }

TILED(matmul_tiled_32x32_f32, 32, 32, 64)
TILED(matmul_tiled_64x64_f32, 64, 64, 32)
TILED(matmul_tiled_32x64_f32, 32, 64, 32)

// Tiled products in two passes, for when there are too few blocks of outputs to
// fill the GPU, as in a weight's gradient, [m, k]·[k, n] with long k and small m
// and n.
//
// The first pass gives each block a BM x BN block of outputs of one batch
// matrix, as matmul_tiled does, and a run of `run` chunks of CHUNK along k. It
// sums each chunk as matmul_tiled does and writes the chunk sums to scratch,
// [batch][chunk][row][column]. The second pass, matmul_chunk_sums_f32, adds each
// output's chunk sums in chunk order from zero.
//
// Bit-identical to matmul_tiled, and so to matmul_split_f32: the same chunk
// sums, added in the same order, with the same exact zero add at the end.
template <unsigned int BM, unsigned int BN, unsigned int BK, unsigned int TM, unsigned int TN>
__device__ void matmul_tiled_chunks(const float* a, const float* b, float* scratch, const View& va, const View& vb,
                                    unsigned int m, unsigned int k, unsigned int n, unsigned int run) {
    constexpr unsigned int TX = BN / TN, TY = BM / TM, THREADS = TX * TY;
    __shared__ float as[BK * (BM + 1)];
    __shared__ float bs[BK * (BN + 1)];
    unsigned int chunks = (k + CHUNK - 1) / CHUNK, runs = (chunks + run - 1) / run;
    unsigned int batch = blockIdx.z / runs, first = blockIdx.z % runs * run;
    unsigned int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    unsigned int t = threadIdx.y * TX + threadIdx.x;
    int pa = offset_of(va, batch * m * k);
    int pb = offset_of(vb, batch * k * n);
    int ra = stride_of(va, va.ndim - 2), sa = stride_of(va, va.ndim - 1);
    int sb = stride_of(vb, vb.ndim - 2), cb = stride_of(vb, vb.ndim - 1);
    float partial[TM][TN];
    for (unsigned int i = 0; i < TM; i++) {
        for (unsigned int j = 0; j < TN; j++) partial[i][j] = 0.0f;
    }
    unsigned int stop = min(k, (first + run) * CHUNK);
    for (unsigned int k0 = first * CHUNK; k0 < stop; k0 += BK) {
        unsigned int width = min(BK, k - k0);
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
            unsigned int c = k0 / CHUNK;
            for (unsigned int i = 0; i < TM; i++) {
                unsigned int row = row0 + threadIdx.y + i * TY;
                for (unsigned int j = 0; j < TN; j++) {
                    unsigned int col = col0 + threadIdx.x + j * TX;
                    if (row < m && col < n) scratch[((batch * chunks + c) * m + row) * n + col] = partial[i][j];
                    partial[i][j] = 0.0f;
                }
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void matmul_tiled_chunks_64x64_f32(const float* a, const float* b, float* scratch, View va, View vb,
                                                         unsigned int m, unsigned int k, unsigned int n, unsigned int run) {
    matmul_tiled_chunks<64, 64, 32, 4, 4>(a, b, scratch, va, vb, m, k, n, run);
}

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

// Visits every element of the row, thread t taking elements t, t + threads and
// so on, so that a warp reads and writes neighbouring elements together. Each
// element's result depends only on its own inputs and the row's totals, so the
// order changes no bits. It follows a ROW_SUM, whose last barrier makes every
// thread's writes to the row visible to the others.
#define OWN_ELEMENTS(body) \
    for (unsigned int s = t; s < n; s += threads) { body }

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

// Returns element `s` of a row of `x · scale + mask`, from the row's starts and
// strides in `a` and `m`: the scale when `flags` has bit 1, the mask when it has
// bit 2, each operation its own statement, rounded as its own op would round it.
__device__ float scaled_at(const float* a, int start, int stride, const float* m, int mstart, int mstride,
                           unsigned int s, float scale, unsigned int flags) {
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
extern "C" __global__ void softmax_scaled_f32(const float* a, const float* m, float* dst, View v, View vm,
                                              unsigned int n, float scale, unsigned int flags) {
    __shared__ float partial[REDUCE_THREADS];
    __shared__ float shared;
    unsigned int row = blockIdx.x, t = threadIdx.x, threads = blockDim.x;
    unsigned int base = row * n;
    int start = offset_of(v, base), stride = run_stride(v);
    int mstart = offset_of(vm, base), mstride = run_stride(vm);
    unsigned int chunks = (n + REDUCE_CHUNK - 1) / REDUCE_CHUNK;
    float top = scaled_at(a, start, stride, m, mstart, mstride, 0, scale, flags);
    for (unsigned int first = 0; first < chunks; first += threads) {
        unsigned int c = first + t;
        if (c < chunks) {
            unsigned int stop = min(n, (c + 1) * REDUCE_CHUNK);
            float acc = scaled_at(a, start, stride, m, mstart, mstride, c * REDUCE_CHUNK, scale, flags);
            for (unsigned int s = c * REDUCE_CHUNK + 1; s < stop; s++) {
                float x = scaled_at(a, start, stride, m, mstart, mstride, s, scale, flags);
                acc = x > acc ? x : acc;
            }
            partial[t] = acc;
        }
        __syncthreads();
        if (t == 0) {
            unsigned int k_end = min(threads, chunks - first);
            for (unsigned int k = 0; k < k_end; k++) top = partial[k] > top ? partial[k] : top;
        }
        __syncthreads();
    }
    if (t == 0) shared = top;
    __syncthreads();
    top = shared;
    __syncthreads();
    float total;
    ROW_SUM(total, float e = expf(scaled_at(a, start, stride, m, mstart, mstride, s, scale, flags) - top); dst[base + s] = e; acc = acc + e;)
    OWN_ELEMENTS(dst[base + s] = dst[base + s] / total;)
}

// The gradient of a softmax's input, per row: the sum of g · y, in chunk order
// as sum_f32 adds; then y · (g - sum), times the scale when `flags` has bit 1.
extern "C" __global__ void softmax_grad_f32(const float* g, const float* y, float* dst, View vg, View vy,
                                            unsigned int n, float scale, unsigned int flags) {
    __shared__ float partial[REDUCE_THREADS];
    __shared__ float shared;
    unsigned int row = blockIdx.x, t = threadIdx.x, threads = blockDim.x;
    unsigned int base = row * n;
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
// it launches as softmax_grad_f32 does.
extern "C" __global__ void log_softmax_grad_f32(const float* g, const float* y, float* dst, View vg, View vy,
                                                unsigned int n, float scale, unsigned int flags) {
    __shared__ float partial[REDUCE_THREADS];
    __shared__ float shared;
    unsigned int row = blockIdx.x, t = threadIdx.x, threads = blockDim.x;
    unsigned int base = row * n;
    int gstart = offset_of(vg, base), gstride = run_stride(vg);
    int ystart = offset_of(vy, base), ystride = run_stride(vy);
    float total;
    ROW_SUM(total, acc = acc + g[gstart + int(s) * gstride];)
    OWN_ELEMENTS(
        float e = expf(y[ystart + int(s) * ystride]);
        float p = e * total;
        dst[base + s] = g[gstart + int(s) * gstride] - p;)
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

// Layer norm's gradients, per row, with the bits of the composite of primitives:
// mean = Σx / n, centred = x - mean, σ = sqrt(Σcentred² / n + epsilon) and
// x̂ = centred / σ, each sum in chunk order as sum_f32 adds. When `with_g` is 0,
// the row of x̂; otherwise, with ĝ = g · w, ((ĝ - Σĝ / n) - x̂ · Σ(ĝ · x̂) / n) / σ.
extern "C" __global__ void layernorm_grad_f32(const float* g, const float* x, const float* w, float* dst,
                                              View vg, View vx, View vw, unsigned int n, float epsilon,
                                              unsigned int with_g) {
    __shared__ float partial[REDUCE_THREADS];
    __shared__ float shared;
    unsigned int row = blockIdx.x, t = threadIdx.x, threads = blockDim.x;
    unsigned int base = row * n;
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
    float std = sqrtf(shifted);
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

// Root-mean-square normalisation of each row: one block per row of a power of
// two threads, at most NORM_THREADS. Thread t adds the squares of elements t,
// t + threads and so on, so that a warp reads neighbouring elements together;
// the threads' sums are then added in a tree, each element divided by
// √(Σx²/n + ε) and scaled by the weight.
extern "C" __global__ void rmsnorm_f32(const float* a, const float* w, float* dst, View v, View vw,
                                       unsigned int n, float epsilon) {
    __shared__ float partial[NORM_THREADS];
    unsigned int row = blockIdx.x, t = threadIdx.x, threads = blockDim.x;
    unsigned int base = row * n;
    int start = offset_of(v, base);
    int stride = run_stride(v);
    int wstart = offset_of(vw, base);
    int wstride = run_stride(vw);
    float acc = 0.0f;
    for (unsigned int s = t; s < n; s += threads) {
        float x = a[start + int(s) * stride];
        float sq = x * x;
        acc = acc + sq;
    }
    partial[t] = acc;
    __syncthreads();
    for (unsigned int width = threads / 2; width > 0; width /= 2) {
        if (t < width) partial[t] = partial[t] + partial[t + width];
        __syncthreads();
    }
    float root = sqrtf(partial[0] / float(n) + epsilon);
    for (unsigned int s = t; s < n; s += threads) {
        float scaled = a[start + int(s) * stride] / root;
        dst[base + s] = scaled * w[wstart + int(s) * wstride];
    }
}

// Rotary embeddings: one thread per element, `x · cos + rotate_half(x) · sin`,
// where rotate_half(x) at a column of the first half is the element half a
// row on, times -1, and at one of the second half the element half a row back.
extern "C" __global__ void rope_f32(const float* a, const float* c, const float* sn, float* dst,
                                    View va, View vc, View vs, unsigned int n, unsigned int count) {
    unsigned int i = thread_index();
    if (i >= count) return;
    unsigned int col = i % n, middle = n / 2;
    float partner = a[offset_of(va, col < middle ? i + middle : i - middle)];
    float rotated = col < middle ? partner * -1.0f : partner;
    float first = a[offset_of(va, i)] * c[offset_of(vc, i)];
    float second = rotated * sn[offset_of(vs, i)];
    dst[i] = first + second;
}

// Block formats (Q8_0, Q4_0, Q4_1), as ggml stores them: per block of 32, an
// F16 scale d (and for Q4_1 an F16 minimum m), then the quantised values. Each
// kernel computes what the host's `Quant` computes, operation for operation.
// Halves are converted by hand, as the host's `F16` does, so that NVRTC needs
// no header and the stand-in driver runs the same code.

__device__ float bits_to_float(unsigned int bits) {
    union { unsigned int u; float f; } value;
    value.u = bits;
    return value.f;
}

__device__ unsigned int float_to_bits(float x) {
    union { float f; unsigned int u; } value;
    value.f = x;
    return value.u;
}

// Returns the half with bits `h` as a float, exactly.
__device__ float half_to_float(unsigned int h) {
    unsigned int sign = (h & 0x8000u) << 16, exponent = (h >> 10) & 0x1fu, fraction = h & 0x3ffu;
    if (exponent == 0) {
        float magnitude = float(fraction) * 5.9604645e-8f;
        return sign ? -magnitude : magnitude;
    }
    if (exponent == 31) return bits_to_float(sign | 0x7f800000u | (fraction << 13));
    return bits_to_float(sign | ((exponent + 112) << 23) | (fraction << 13));
}

// Returns x rounded to the nearest half, ties to even, as a float.
__device__ float round_to_half(float x) {
    unsigned int word = float_to_bits(x);
    unsigned int sign = (word >> 16) & 0x8000u, magnitude = word & 0x7fffffffu, h;
    if (magnitude >= 0x7f800000u) h = sign | (magnitude > 0x7f800000u ? 0x7e00u : 0x7c00u);
    else if (magnitude >= 0x477ff000u) h = sign | 0x7c00u;
    else if (magnitude < 0x38800000u) h = sign | (unsigned int)rintf(bits_to_float(magnitude) * 16777216.0f);
    else h = sign | ((magnitude - 0x38000000u + 0xfffu + ((magnitude >> 13) & 1u)) >> 13);
    return half_to_float(h);
}

__device__ float block_half(const unsigned char* p) {
    return half_to_float(unsigned(p[0]) | unsigned(p[1]) << 8);
}

// Returns element j of a row of blocks of `kind` (0 Q8_0, 1 Q4_0, 2 Q4_1)
// and `bytes` bytes each, starting at p.
__device__ float block_value(const unsigned char* p, unsigned int kind, unsigned int bytes, unsigned int j) {
    const unsigned char* block = p + (j / 32) * bytes;
    unsigned int at = j % 32;
    if (kind == 0) return float((signed char)block[2 + at]) * block_half(block);
    unsigned int packed = block[(kind == 1 ? 2 : 4) + at % 16];
    unsigned int q = at < 16 ? (packed & 0x0fu) : (packed >> 4);
    if (kind == 1) return float(int(q) - 8) * block_half(block);
    return block_half(block) * float(q) + block_half(block + 2);
}

// Gather from a block-format table: one thread per output element, the rows
// dequantised as they are read; `offset` is the byte offset of the first row.
extern "C" __global__ void gather_blocks_f32(const unsigned char* table, const int* ids, float* dst, int* errors,
                                             unsigned int kind, unsigned int bytes, unsigned int offset,
                                             View vi, unsigned int width, int rows, unsigned int n) {
    unsigned int i = thread_index();
    if (i >= n) return;
    int id = ids[offset_of(vi, i / width)];
    if (id < 0 || id >= rows) {
        report(errors, id, rows);
        dst[i] = 0.0f;
        return;
    }
    const unsigned char* row = table + offset + (unsigned int)id * (width / 32) * bytes;
    dst[i] = block_value(row, kind, bytes, i % width);
}

// Quantises each block of 32 activations to 8 bits, as the host's
// `quantize_activations` does: one thread per block of the rows read through
// vx, writing the F16-rounded scale, d·Σq rounded to F16, and the values.
extern "C" __global__ void quantize_rows_q8_f32(const float* x, float* scales, float* sums, signed char* qs,
                                                View vx, unsigned int blocks) {
    unsigned int i = thread_index();
    if (i >= blocks) return;
    unsigned int base = i * 32;
    float amax = 0.0f;
    for (unsigned int j = 0; j < 32; j++) amax = fmaxf(amax, fabsf(x[offset_of(vx, base + j)]));
    float d = amax / 127.0f;
    float id = d == 0.0f ? 0.0f : 1.0f / d;
    int total = 0;
    for (unsigned int j = 0; j < 32; j++) {
        signed char q = (signed char)roundf(x[offset_of(vx, base + j)] * id);
        qs[base + j] = q;
        total += q;
    }
    scales[i] = round_to_half(d);
    sums[i] = round_to_half(float(total) * d);
}

// Returns the contribution of one block of a weight row, at p, to an output:
// its integer dot product with the block's quantised activations, qb, times
// both scales, plus m · d·Σq for Q4_1, with the host's expressions, so that
// adding the contributions in block order gives the host's bits.
__device__ float block_contribution(const unsigned char* p, const signed char* qb, float sx, float mx, unsigned int kind) {
    if (kind == 0) {
        int total = 0;
        for (unsigned int j = 0; j < 32; j++) total += int((signed char)p[2 + j]) * int(qb[j]);
        return float(total) * (block_half(p) * sx);
    }
    int low = 0, high = 0;
    unsigned int base = kind == 1 ? 2 : 4;
    int bias = kind == 1 ? 8 : 0;
    for (unsigned int j = 0; j < 16; j++) {
        unsigned int packed = p[base + j];
        low += (int(packed & 0x0fu) - bias) * int(qb[j]);
        high += (int(packed >> 4) - bias) * int(qb[j + 16]);
    }
    if (kind == 1) return float(low + high) * block_half(p) * sx;
    return (block_half(p) * sx) * float(low + high) + block_half(p + 2) * mx;
}

// x · wᵀ for a block-format weight: one thread per output, adding its blocks'
// contributions in order. For products with outputs enough to fill the GPU.
extern "C" __global__ void block_matmul_f32(const unsigned char* w, const float* scales, const float* sums,
                                            const signed char* qs, float* dst, const float* bias,
                                            unsigned int kind, unsigned int bytes, unsigned int offset,
                                            unsigned int n, unsigned int k, unsigned int count) {
    unsigned int i = thread_index();
    if (i >= count) return;
    unsigned int blocks = k / 32, row = i / n, col = i % n;
    const unsigned char* weights = w + offset + col * blocks * bytes;
    float sum = 0.0f;
    for (unsigned int block = 0; block < blocks; block++)
        sum += block_contribution(weights + block * bytes, qs + row * k + block * 32, scales[row * blocks + block],
                                  sums[row * blocks + block], kind);
    dst[i] = with_bias(sum, bias, col);
}

// Returns the 4 bytes at p, which is 2-byte aligned, as one little-endian word.
__device__ int word_at(const unsigned char* p) {
    const unsigned short* h = (const unsigned short*)p;
    return int(unsigned(h[0]) | unsigned(h[1]) << 16);
}

// Returns block_contribution's value for the block at p, which is 2-byte
// aligned, from its quantised activations as 8 words, a: the same integer dot
// product, four products an instruction, and the same expressions after it.
__device__ float wide_contribution(const unsigned char* p, const int* a, float sx, float mx, unsigned int kind) {
    int total = 0;
    if (kind == 0) {
        for (unsigned int j = 0; j < 8; j++) total = __dp4a(word_at(p + 2 + 4 * j), a[j], total);
        return float(total) * (block_half(p) * sx);
    }
    const unsigned char* packed = p + (kind == 1 ? 2 : 4);
    for (unsigned int j = 0; j < 4; j++) {
        int word = word_at(packed + 4 * j);
        total = __dp4a(word & 0x0f0f0f0f, a[j], total);
        total = __dp4a((word >> 4) & 0x0f0f0f0f, a[4 + j], total);
    }
    if (kind == 1) {
        int sum = 0;
        for (unsigned int j = 0; j < 8; j++) sum = __dp4a(0x01010101, a[j], sum);
        return float(total - 8 * sum) * block_half(p) * sx;
    }
    return (block_half(p) * sx) * float(total) + block_half(p + 2) * mx;
}

// x · wᵀ for a block-format weight with few outputs: `per` consecutive outputs
// (at most LANE_OUTPUTS) per warp, `blockDim.x / 32` warps a block. Lane l
// computes the contributions of blocks l, l + 32 and so on to each of its
// warp's outputs, reading the weights two bytes at a time and the quantised
// activations four, once for the outputs that share a row. For each 32 blocks,
// the lanes put their contributions in shared memory, and lane r adds those of
// output r in block order, as the host does, so the result has the host's
// bits. Outputs past the last compute the last one and write nothing. The
// weights must start at an even byte offset.
extern "C" __global__ void block_matmul_lanes_f32(const unsigned char* w, const float* scales, const float* sums,
                                                  const signed char* qs, float* dst, const float* bias,
                                                  unsigned int kind, unsigned int bytes, unsigned int offset,
                                                  unsigned int n, unsigned int k, unsigned int count, unsigned int per) {
    __shared__ float parts[BLOCK_WARPS * LANE_OUTPUTS * LANE_STRIDE];
    unsigned int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    unsigned int first = (blockIdx.x * (blockDim.x / 32) + warp) * per;
    unsigned int blocks = k / 32;
    float* mine = parts + warp * LANE_OUTPUTS * LANE_STRIDE;
    unsigned int lead = min(first, count - 1) / n;
    float sum = 0.0f;
    for (unsigned int start = 0; start < blocks; start += 32) {
        unsigned int block = start + lane;
        if (block < blocks) {
            int a[8];
            const int* lead_words = (const int*)(qs + lead * k + block * 32);
            for (unsigned int j = 0; j < 8; j++) a[j] = lead_words[j];
            for (unsigned int r = 0; r < per; r++) {
                unsigned int i = min(first + r, count - 1);
                unsigned int row = i / n, col = i % n;
                const unsigned char* p = w + offset + (col * blocks + block) * bytes;
                unsigned int at = row * blocks + block;
                if (row == lead) {
                    mine[r * LANE_STRIDE + lane] = wide_contribution(p, a, scales[at], sums[at], kind);
                } else {
                    int own[8];
                    const int* words = (const int*)(qs + row * k + block * 32);
                    for (unsigned int j = 0; j < 8; j++) own[j] = words[j];
                    mine[r * LANE_STRIDE + lane] = wide_contribution(p, own, scales[at], sums[at], kind);
                }
            }
        }
        __syncwarp();
        if (lane < per) {
            unsigned int run = blocks - start < 32 ? blocks - start : 32;
            for (unsigned int j = 0; j < run; j++) sum += mine[lane * LANE_STRIDE + j];
        }
        __syncwarp();
    }
    unsigned int output = first + lane;
    if (lane < per && output < count) dst[output] = with_bias(sum, bias, output % n);
}
