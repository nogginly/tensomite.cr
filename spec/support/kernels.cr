# Kernels the specs define; scripts/cuda_stub also compiles them for its emulated GPU.
Tensomite.kernel(:spec_axpb) { |x, a, b| x * a + b }
Tensomite.kernel(:spec_mixed) { |x, y| (-x / (y * y + 1)).exp + (x * x + 1).log - (y * y + 1).sqrt }
Tensomite.kernel(:spec_folded) { |x| x * (Math::PI / 2) - 1 / 3 }
