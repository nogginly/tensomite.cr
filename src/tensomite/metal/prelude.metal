// What every Metal kernel source starts with: the view struct and offset_of,
// which reads a source through its view exactly as View#offset_of does, so any
// view the host kernels accept works on the GPU too.
#include <metal_stdlib>
using namespace metal;

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
static int offset_of(constant View& v, int flat) {
    if (v.dense) return v.offset + flat;
    int pos = v.offset;
    for (int d = v.ndim - 1; d >= 0; d--) {
        int dim = v.shape[d];
        pos += (flat % dim) * v.strides[d];
        flat /= dim;
    }
    return pos;
}

// Returns an output in column *col* of a matrix product, *out*, plus its bias,
// bias[col], when *biased*.
static float with_bias(float out, device const float* bias, uint col, uint biased) {
    return biased ? out + bias[col] : out;
}

// Returns the step between the positions of consecutive elements along the last
// dimension.
static int run_stride(constant View& v) {
    return v.ndim > 0 ? v.strides[v.ndim - 1] : 0;
}
