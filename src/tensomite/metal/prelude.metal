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

