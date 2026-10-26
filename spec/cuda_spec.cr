require "./spec_helper"

{% if flag?(:cuda) %}
  private SOURCE = <<-CUDA
    extern "C" __global__ void add_f32(const float* a, const float* b, float* out, int n) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) out[i] = a[i] + b[i];
    }

    extern "C" __global__ void multiply_add_f32(const float* a, const float* b, const float* c, float* out) {
      out[0] = a[0] * b[0] + c[0];
    }
    CUDA

  # Returns *values* as the bytes of `Float32`s.
  private def bytes_of(values : Array(Float32)) : Bytes
    Bytes.new(values.to_unsafe.as(UInt8*), values.size * 4).dup
  end

  private def floats_of(bytes : Bytes) : Array(Float32)
    Array(Float32).new(bytes.size // 4) { |i| bytes.to_unsafe.as(Float32*)[i] }
  end

  # NVRTC needs no GPU, so these run wherever the library is.
  describe "CUDA's compiler, NVRTC" do
    it "compiles CUDA C++ to PTX for an architecture it supports" do
      CUDA.nvrtc_version[0].should be >= 12
      arch = CUDA.supported_archs.min
      ptx = CUDA.compile(SOURCE, arch)
      ptx.should contain ".entry add_f32"
      ptx.should contain ".target sm_#{arch}"
    end

    it "keeps a * b + c as a multiply and an add, not one fused multiply-add" do
      ptx = CUDA.compile(SOURCE, CUDA.supported_archs.min)
      body = ptx[ptx.index!(".entry multiply_add_f32")..]
      body.should contain "mul.rn.f32"
      body.should contain "add.rn.f32"
      body.should_not contain "fma"
    end

    it "passes each kernel argument as a pointer to a value of its own type" do
      params = CUDA.kernel_params(0x1234_5678_9abc_u64, 7, 2.5f32)
      params[0].as(UInt64*).value.should eq 0x1234_5678_9abc_u64
      params[1].as(Int32*).value.should eq 7
      params[2].as(Float32*).value.should eq 2.5f32
    end

    it "reports compile errors with NVRTC's log" do
      expect_raises(CUDA::CompileError, /broken.cu.*error/m) { CUDA.compile("this is not CUDA", CUDA.supported_archs.min, "broken.cu") }
    end
  end

  # The stand-in libcuda of scripts/cuda_stub has a GPU that runs no kernels.
  private RUNS_KERNELS = CUDA.available? && !CUDA.gpus.first.name.includes?("host stub")

  describe "CUDA, the device" do
    pending "no CUDA GPU on this machine" unless CUDA.available?

    if CUDA.available?
      cuda = CUDA.default

      it "never exposes its memory to the host" do
        cuda.map(Tensor.from(f32(1, 2), [2], cuda).node.buffer!).should be_nil
      end

      it "runs ops without CUDA kernels on the host, moving data through counted transfers" do
        uploads, downloads = cuda.uploads, cuda.downloads
        t = Tensor.from(f32(1, 2, 3), [3], cuda)
        (t + t).to_a(Float32).should eq f32(2, 4, 6)
        (cuda.uploads - uploads).should eq 1
        (cuda.downloads - downloads).should eq 1
      end

      it "downloads a buffer once for host kernels, until the GPU writes it again" do
        t = Tensor.from(f32(1, 2, 3), [3], cuda)
        buffer = t.node.buffer!.as(CUDABuffer)
        buffer.current_copy.should be_nil
        ((t * 2) + (t * 3)).to_a(Float32).should eq f32(5, 10, 15)
        buffer.current_copy.should_not be_nil
        cuda.upload(buffer, Bytes.new(12))
        buffer.current_copy.should be_nil
        (t + 1).to_a(Float32).should eq f32(1, 1, 1)
      end

      it "rejects buffers belonging to another device" do
        expect_raises(DeviceError) { cuda.download(Tensor.from(f32(1), [1]).node.buffer!, Bytes.new(4)) }
      end

      it "copies tensors loaded from a file to the GPU" do
        path = File.tempname("tensomite", ".safetensors")
        SafeTensors.write(path, {"w" => Tensor.from(f32(1, 2, 3, 4), [2, 2])})
        w = SafeTensors.open(path)["w", cuda]
        w.node.buffer!.should be_a CUDABuffer
        w.matmul(w).to_a(Float32).should eq f32(7, 10, 15, 22)
      end

      it "reuses the device memory of collected buffers" do
        hits = cuda.buffer_pool.hits
        10.times do
          100.times { (Tensor.from(Array.new(256, 1f32), [256], cuda) * 2).realize }
          GC.collect
          cuda.sync
          break if cuda.buffer_pool.hits > hits
        end
        cuda.buffer_pool.hits.should be > hits
      end
    end
  end

  describe "CUDA kernels" do
    pending "no CUDA GPU that runs kernels on this machine" unless RUNS_KERNELS

    if RUNS_KERNELS
      gpu = CUDA.gpus.first
      context = CUDA::Context.new(gpu)
      kernels = context.load(CUDA.compile(SOURCE, CUDA.target_arch(gpu)))

      it "describes its GPUs and versions" do
        gpu.name.should_not be_empty
        gpu.memory.should be > 0
        gpu.arch.should be >= 50
        CUDA.driver_version[0].should be >= 12
        CUDA.nvrtc_version[0].should be >= 12
      end

      it "copies to and from the GPU and runs a compiled kernel" do
        n = 100_000
        a = Array(Float32).new(n) { |i| i.to_f32 }
        b = Array(Float32).new(n) { |i| (2 * i).to_f32 }
        pointers = Array.new(3) { context.alloc(4 * n) }
        context.upload(pointers[0], bytes_of(a))
        context.upload(pointers[1], bytes_of(b))
        context.launch(kernels["add_f32"], (n + 255) // 256, 256, pointers[0], pointers[1], pointers[2], n)
        context.synchronize
        result = Bytes.new(4 * n)
        context.download(pointers[2], result)
        floats_of(result).should eq Array(Float32).new(n) { |i| (3 * i).to_f32 }
        pointers.each { |pointer| context.free(pointer) }
      end

      it "rounds a * b + c twice, as the host does, not as one fused multiply-add" do
        # (1 + 2^-23)² = 1 + 2^-22 + 2^-46: rounded, the 2^-46 is lost and adding
        # c gives 0; fused, it survives.
        a = 1f32 + Float32::EPSILON
        c = -(1f32 + 2 * Float32::EPSILON)
        (a * a + c).should eq 0f32
        pointers = Array.new(4) { context.alloc(4) }
        [a, a, c].each_with_index { |value, i| context.upload(pointers[i], bytes_of([value])) }
        context.launch(kernels["multiply_add_f32"], 1, 1, pointers[0], pointers[1], pointers[2], pointers[3])
        result = Bytes.new(4)
        context.download(pointers[3], result)
        floats_of(result).should eq [0f32]
        pointers.each { |pointer| context.free(pointer) }
      end
    end
  end
{% end %}
