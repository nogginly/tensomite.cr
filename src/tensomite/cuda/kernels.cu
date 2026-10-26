// Hand-written kernels for the CUDA device, compiled by NVRTC with fused
// multiply-adds off, so +, -, * and comparisons give the same bits as the host
// kernels. Each is a port of the Metal kernel of the same name.

#define MAX_DIMS 8

// A view as the host describes it: offset, then shape and strides per dimension.
struct View {
    int ndim;
    int offset;
    int shape[MAX_DIMS];
    int strides[MAX_DIMS];
    int dense;  // 1 when the strides are row-major: element i is at offset + i
};

// Returns the storage position of logical element `flat` of `v`, peeling indices
// off from the last dimension, or directly when the view is dense.
__device__ int offset_of(const View& v, int flat) {
    if (v.dense) return v.offset + flat;
    int pos = v.offset;
    for (int d = v.ndim - 1; d >= 0; d--) {
        int dim = v.shape[d];
        pos += (flat % dim) * v.strides[d];
        flat /= dim;
    }
    return pos;
}

// Returns the step between the positions of consecutive elements along the last
// dimension.
__device__ int run_stride(const View& v) {
    return v.ndim > 0 ? v.strides[v.ndim - 1] : 0;
}

__device__ unsigned int thread_index() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

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
