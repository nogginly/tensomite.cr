# Prints the CUDA driver and NVRTC versions and each GPU the driver sees, then
# compiles and runs a kernel on the first GPU to check the whole path.
#
#     LIBRARY_PATH=/usr/local/cuda-13.1/targets/sbsa-linux/lib crystal run -Dcuda examples/cuda_info.cr
require "../src/tensomite/cuda"

include Tensomite

abort "CUDA is not available: no driver or no GPU" unless CUDA.available?

puts "driver supports CUDA #{CUDA.driver_version.join('.')}, NVRTC #{CUDA.nvrtc_version.join('.')}"
puts "NVRTC compiles for #{CUDA.supported_archs.join(", ")}"
CUDA.gpus.each do |gpu|
  memory = "#{(gpu.memory / 2**30).round(1)} GiB #{gpu.integrated ? "shared with the CPU" : "on the card"}"
  puts "GPU #{gpu.ordinal}: #{gpu.name}, compute #{gpu.compute_capability.join('.')}, #{gpu.multiprocessors} multiprocessors, #{memory}"
end

source = <<-CUDA
  extern "C" __global__ void square_f32(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * x[i];
  }
  CUDA

gpu = CUDA.gpus.first
arch = CUDA.target_arch(gpu)
context = CUDA::Context.new(gpu)
kernels = context.load(CUDA.compile(source, arch))
n = 1 << 20
values = Array(Float32).new(n) { |i| (i % 1000).to_f32 }
buffer = context.alloc(4 * n)
context.upload(buffer, Bytes.new(values.to_unsafe.as(UInt8*), 4 * n))
context.launch(kernels["square_f32"], (n + 255) // 256, 256, buffer, n)
result = Array(Float32).new(n, 0f32)
context.download(buffer, Bytes.new(result.to_unsafe.as(UInt8*), 4 * n))
context.free(buffer)
correct = result.each_with_index.all? { |value, i| value == ((i % 1000) ** 2).to_f32 }
puts "squared #{n} floats on GPU 0 with PTX for compute_#{arch}: #{correct ? "correct" : "WRONG"}"
exit(correct ? 0 : 1)
