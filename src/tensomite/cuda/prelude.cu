// What every CUDA kernel source starts with: the view struct and offset_of,
// which reads a source through its view exactly as View#offset_of does, so any
// view the host kernels accept works on the GPU too.

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

// Returns output i of a matrix product, *out*, plus its bias: c read through vc,
// a view of the output's shape, when the product has one (c is not null).
__device__ float with_bias(float out, const float* c, const View& vc, unsigned int i) {
    return c ? out + c[offset_of(vc, int(i))] : out;
}

__device__ unsigned int thread_index() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}
