# Prints the CUDA C++ of every kernel defined with `Tensomite.kernel`, tensomite's
# own and the specs', for scripts/cuda_stub/setup.sh to compile as host C++.
require "../../src/tensomite"
require "../../spec/support/kernels"

{% for kernel in Tensomite::Kernel.all_subclasses %}
  {% unless kernel.abstract? %}
    puts {{ kernel }}::INSTANCE.cuda
  {% end %}
{% end %}
