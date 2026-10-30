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

// A view is a kernel argument, and indexing an argument by a value known only at
// run time makes the compiler copy it to each thread's local memory first. So
// these helpers loop over every possible dimension, which unrolls into indices
// known when compiling, and skip the dimensions the view does not have.

// Returns the storage position of logical element `flat` of `v`, peeling indices
// off from the last dimension, or directly when the view is dense.
__device__ int offset_of(const View& v, int flat) {
    if (v.dense) return v.offset + flat;
    int pos = v.offset;
#pragma unroll
    for (int d = MAX_DIMS - 1; d >= 0; d--) {
        if (d < v.ndim) {
            int dim = v.shape[d];
            pos += (flat % dim) * v.strides[d];
            flat /= dim;
        }
    }
    return pos;
}

// Returns the stride of dimension `d` of `v`, or 0 when it has none.
__device__ int stride_of(const View& v, int d) {
    int stride = 0;
#pragma unroll
    for (int i = 0; i < MAX_DIMS; i++) {
        if (i == d) stride = v.strides[i];
    }
    return stride;
}

// Returns the step between the positions of consecutive elements along the last
// dimension.
__device__ int run_stride(const View& v) {
    return stride_of(v, v.ndim - 1);
}

// Returns an output in column *col* of a matrix product, *out*, plus its bias,
// bias[col], when the product has one (bias is not null).
__device__ float with_bias(float out, const float* bias, unsigned int col) {
    return bias ? out + bias[col] : out;
}

__device__ unsigned int thread_index() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}
