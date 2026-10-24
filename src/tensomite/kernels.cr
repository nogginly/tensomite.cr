# The fused kernels tensomite's own ops use; see `Tensomite.kernel`.

# GELU with the tanh approximation, as `Tensor#gelu`.
Tensomite.kernel(:gelu) { |x| x * 0.5 * (((x + x * x * x * 0.044715) * Math.sqrt(2 / Math::PI)).tanh + 1) }
