# The fused kernels tensomite's own ops use; see `Tensomite.kernel`.

# GELU with the tanh approximation, as `Tensor#gelu`.
Tensomite.kernel(:gelu) { |x| x * 0.5 * (((x + x * x * x * 0.044715) * Math.sqrt(2 / Math::PI)).tanh + 1) }

# AdamW's moments and update, as `AdamW` uses them after its first step: the
# operations of its composite, in the same order.
Tensomite.kernel(:adamw_moment) { |moment, grad, beta, rest| moment * beta + grad * rest }
Tensomite.kernel(:adamw_second_moment) { |moment, grad, beta, rest| moment * beta + grad * grad * rest }
Tensomite.kernel(:adamw_update) do |param, moment, second, decay, m_scale, v_scale, eps, rate|
  param * decay - moment / m_scale / ((second / v_scale).sqrt + eps) * rate
end
