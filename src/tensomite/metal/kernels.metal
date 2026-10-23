// Kernels for the Metal device. Each thread computes one output element.
//
// Sources are read through their views exactly as View#offset_of does, so any
// view the host kernels accept works here too. Compiled with fast math off, so
// +, -, * and comparisons give the same bits as the host kernels.
#include <metal_stdlib>
using namespace metal;

#define MAX_DIMS 8

// A view as the host describes it: offset, then shape and strides per dimension.
struct View {
    int ndim;
    int offset;
    int shape[MAX_DIMS];
    int strides[MAX_DIMS];
};

// Returns the storage position of logical element `flat` of `v`, peeling indices
// off from the last dimension.
static int offset_of(constant View& v, int flat) {
    int pos = v.offset;
    for (int d = v.ndim - 1; d >= 0; d--) {
        int dim = v.shape[d];
        pos += (flat % dim) * v.strides[d];
        flat /= dim;
    }
    return pos;
}

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

// Folds each output's run of `run` consecutive logical indices of `v` in index
// order, as the host does: Sum from zero, Max from the first element.
kernel void sum_f32(device const float* a [[buffer(0)]],
                    device float* dst [[buffer(1)]],
                    constant View& v [[buffer(2)]],
                    constant uint& run [[buffer(3)]],
                    constant uint& count [[buffer(4)]],
                    uint row [[thread_position_in_grid]]) {
    if (row >= count) return;
    float acc = 0.0f;
    for (uint j = 0; j < run; j++) acc += a[offset_of(v, row * run + j)];
    dst[row] = acc;
}

kernel void max_f32(device const float* a [[buffer(0)]],
                    device float* dst [[buffer(1)]],
                    constant View& v [[buffer(2)]],
                    constant uint& run [[buffer(3)]],
                    constant uint& count [[buffer(4)]],
                    uint row [[thread_position_in_grid]]) {
    if (row >= count) return;
    float acc = a[offset_of(v, row * run)];
    for (uint j = 1; j < run; j++) {
        float x = a[offset_of(v, row * run + j)];
        acc = x > acc ? x : acc;
    }
    dst[row] = acc;
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
