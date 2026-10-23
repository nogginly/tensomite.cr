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
