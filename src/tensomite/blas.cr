require "../tensomite"

# Opt-in BLAS support: `require "tensomite/blas"` links Accelerate on macOS and
# OpenBLAS elsewhere, and adds the `Tensomite::BLAS` device.
module Tensomite
  {% if flag?(:darwin) %}
    @[Link(framework: "Accelerate")]
  {% else %}
    @[Link("openblas")]
  {% end %}
  lib LibBLAS
    fun cblas_sgemm(order : Int32, trans_a : Int32, trans_b : Int32, m : Int32, n : Int32, k : Int32,
                    alpha : Float32, a : Float32*, lda : Int32, b : Float32*, ldb : Int32,
                    beta : Float32, c : Float32*, ldc : Int32) : Void
  end

  # Host kernels on host-visible memory, with `F32` matrix products computed by the
  # platform BLAS (`cblas_sgemm`). Every other op, and `I32` matmul, runs the host
  # kernels.
  #
  # Vendor BLAS libraries choose their blocking by shape, so results are not
  # shape-invariant.
  class BLAS < HostDevice
    private ROW_MAJOR = 101
    private NO_TRANS  = 111
    private TRANS     = 112

    # Returns the shared BLAS device.
    class_getter default : BLAS { BLAS.new }

    def map(buffer : Buffer) : Bytes?
      own(buffer).bytes
    end

    def shape_invariant? : Bool
      false
    end

    def execute(node : Node, output : Buffer) : Nil
      if node.op == Op::MatMul && node.dtype.f32?
        sgemm(node, own(output).bytes.unsafe_slice_of(Float32))
      else
        super
      end
    end

    # Computes each batch matrix with one `cblas_sgemm` call.
    #
    # 1. Describes each operand's batch matrix to BLAS as a pointer, a transpose
    #    flag and a leading dimension, copying it only when its strides allow neither.
    # 2. Writes `[m, n]` results contiguously into *dst*, batch after batch.
    private def sgemm(node : Node, dst : Slice(Float32)) : Nil
      a, b = node.srcs
      m, k, n = a.shape[-2], a.shape[-1], b.shape[-1]
      return if dst.empty?
      return dst.fill(0f32) if k == 0
      xs, ys = elements(Float32, a), elements(Float32, b)
      (dst.size // (m * n)).times do |batch|
        pa, trans_a, lda = operand(a.view, xs, batch, m, k)
        pb, trans_b, ldb = operand(b.view, ys, batch, k, n)
        LibBLAS.cblas_sgemm(ROW_MAJOR, trans_a, trans_b, m, n, k,
          1f32, pa, lda, pb, ldb, 0f32, dst.to_unsafe + batch * m * n, n)
      end
    end

    # Returns the pointer, transpose flag and leading dimension of matrix *batch*
    # of *view*, whose last two dimensions are `[rows, cols]`.
    #
    # Row-major strides (`[ld, 1]`) need no transpose; column-major strides
    # (`[1, ld]`) are the transpose of a row-major matrix. Anything else, such as a
    # broadcast dimension with stride 0, is copied into a packed row-major matrix.
    private def operand(view : View, xs : Slice(Float32), batch : Int32, rows : Int32, cols : Int32) : {Pointer(Float32), Int32, Int32}
      base = view.offset_of(batch * rows * cols)
      row_stride, col_stride = view.strides[-2], view.strides[-1]
      row_ld = rows == 1 ? cols : row_stride
      col_ld = cols == 1 ? rows : col_stride
      if (cols == 1 || col_stride == 1) && row_ld >= Math.max(1, cols)
        {xs.to_unsafe + base, NO_TRANS, row_ld}
      elsif (rows == 1 || row_stride == 1) && col_ld >= Math.max(1, rows)
        {xs.to_unsafe + base, TRANS, col_ld}
      else
        packed = Slice(Float32).new(rows * cols) { |i| xs[view.offset_of(batch * rows * cols + i)] }
        {packed.to_unsafe, NO_TRANS, cols}
      end
    end
  end
end
