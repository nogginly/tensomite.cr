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

  # The stand-in libcuda of scripts/cuda_stub has a GPU in host memory, which
  # runs the device's own kernels compiled as host C++, or none, but never PTX.
  private REAL_GPU = CUDA.available? && !CUDA.gpus.first.name.includes?("host stub")

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

      # I32 ops have no CUDA kernels, so they run on the host.
      it "downloads a buffer once for host kernels, until the GPU writes it again" do
        t = Tensor.from([1, 2, 3], [3], cuda)
        buffer = t.node.buffer!.as(CUDABuffer)
        buffer.current_copy.should be_nil
        ((t * 2) + (t * 3)).to_a(Int32).should eq [5, 10, 15]
        buffer.current_copy.should_not be_nil
        cuda.upload(buffer, Bytes.new(12))
        buffer.current_copy.should be_nil
        (t + 1).to_a(Int32).should eq [1, 1, 1]
      end

      it "runs ported ops on the GPU when it can load kernels, and the rest on the host" do
        gpu, host = cuda.gpu_ops, cuda.host_ops
        a = Tensor.from(f32(1, 2, 3, 4), [2, 2], cuda)
        # a + a = [[2, 4], [6, 8]], summed by rows = [6, 14].
        (a + a).sum(1).to_a(Float32).should eq f32(6, 14)
        (Tensor.from([1, 2], [2], cuda) + Tensor.from([3, 4], [2], cuda)).to_a(Int32).should eq [4, 6]
        if cuda.kernels
          {cuda.gpu_ops - gpu, cuda.host_ops - host}.should eq({2, 1})
        else
          {cuda.gpu_ops - gpu, cuda.host_ops - host}.should eq({0, 3})
        end
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

  describe "CUDA matmul and row ops" do
    pending "no CUDA GPU that runs kernels on this machine" unless CUDA.available? && CUDA.default.kernels

    if CUDA.available? && CUDA.default.kernels
      cuda = CUDA.default
      rng = Random.new(5)
      xs = Array(Float32).new(3 * 200) { rng.rand(-1f32...1f32) }
      ws = Array(Float32).new(200 * 70) { rng.rand(-1f32...1f32) }
      product = ->(mode : CUDA::MatMulMode) do
        cuda.matmul_mode = mode
        Tensor.from(xs, [3, 200], cuda).matmul(Tensor.from(ws, [200, 70], cuda)).to_a(Float32)
  ensure
    cuda.matmul_mode = CUDA::MatMulMode::Auto
      end
      expected = Tensor.from(xs, [3, 200]).matmul(Tensor.from(ws, [200, 70])).to_a(Float32)

      it "is bit-identical to the host when serial" do
        product.call(CUDA::MatMulMode::Serial).should eq expected
      end

      it "keeps the bits of each output when k is padded with zeros" do
        # Prefill and decode differ only by masked positions, which add zeros
        # to k; chunks at fixed offsets make that change nothing.
        padded = xs.each_slice(200).to_a.flat_map { |row| row + Array.new(77, 0f32) }
        padded_ws = ws + Array.new(77 * 70, 0f32)
        split = product.call(CUDA::MatMulMode::Split)
        Tensor.from(padded, [3, 277], cuda).matmul(Tensor.from(padded_ws, [277, 70], cuda)).to_a(Float32).should eq split
      end

      # Returns a·b under *mode* and *tile*; a is [*batch, m, k], and b is
      # [*batch, k, n], or [*batch, n, k] transposed when *b_transposed*.
      tiled_product = ->(mode : CUDA::MatMulMode, tile : CUDA::MatMulTile, a_shape : Array(Int32), b_shape : Array(Int32), b_transposed : Bool) do
        cuda.matmul_mode, cuda.matmul_tile = mode, tile
        rng = Random.new(a_shape.sum + b_shape.sum)
        a = Tensor.from(Array(Float32).new(a_shape.product) { rng.rand(-1f32...1f32) }, a_shape, cuda)
        b = Tensor.from(Array(Float32).new(b_shape.product) { rng.rand(-1f32...1f32) }, b_shape, cuda)
        a.matmul(b_transposed ? b.transpose : b).to_a(Float32)
  ensure
    cuda.matmul_mode, cuda.matmul_tile = CUDA::MatMulMode::Auto, CUDA::MatMulTile::T64x64
      end
      shapes = [
        {[3, 200], [200, 70], false}, {[70, 129], [129, 65], false}, {[64, 64], [64, 64], false},
        {[2, 37, 577], [2, 577, 33], false}, {[33, 100], [97, 100], true}, {[5, 1], [1, 3], false},
      ]

      it "gives split's bits when tiled, with every tile, batched and transposed" do
        shapes.each do |(a_shape, b_shape, b_transposed)|
          split = tiled_product.call(CUDA::MatMulMode::Split, CUDA::MatMulTile::T64x64, a_shape, b_shape, b_transposed)
          CUDA::MatMulTile.values.each do |tile|
            tiled_product.call(CUDA::MatMulMode::Tiled, tile, a_shape, b_shape, b_transposed).should eq split
          end
        end
      end

      it "gives split's bits in two passes, with few rows or many, batched and transposed" do
        (shapes + [{[1, 768], [300, 768], true}, {[2, 3, 1, 129], [2, 3, 129, 200], false}, {[1, 0], [0, 5], false}]).each do |(a_shape, b_shape, b_transposed)|
          split = tiled_product.call(CUDA::MatMulMode::Split, CUDA::MatMulTile::T64x64, a_shape, b_shape, b_transposed)
          tiled_product.call(CUDA::MatMulMode::TwoPass, CUDA::MatMulTile::T64x64, a_shape, b_shape, b_transposed).should eq split
        end
      end

      it "gives a tiled prefill the bits of a decode" do
        model = GPT2.load("#{__DIR__}/fixtures/tiny_gpt2/model.safetensors", heads: 4, device: cuda)
        ids = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
        cuda.matmul_mode = CUDA::MatMulMode::Tiled
        prefill = model.forward(ids, model.new_cache).to_a(Float32)
        cuda.matmul_mode = CUDA::MatMulMode::Split
        decode = model.new_cache
        ids.flat_map { |id| model.forward([id], decode).to_a(Float32) }.should eq prefill
      ensure
        cuda.matmul_mode = CUDA::MatMulMode::Auto
      end

      it "gives a softmax row the same bits when it is padded with -inf" do
        row = Array(Float32).new(300) { |i| ((i * 37) % 101 - 50) / 10f32 }
        padded = row + Array.new(200, -Float32::INFINITY)
        short = Tensor.from(row, [1, 300], cuda).softmax.to_a(Float32)
        Tensor.from(padded, [1, 500], cuda).softmax.to_a(Float32)[0, 300].should eq short
      end

      it "profiles launches, transfers and GPU time per op" do
        a = Tensor.from(f32(1, 2, 3, 4), [2, 2], cuda)
        profile = CUDA::Profile.new(per_kernel: true, shapes: true)
        cuda.profile = profile
        # (a + 1) * a = [2, 6, 12, 20], which sums to 40.
        ((a + 1) * a).sum.to_a(Float32).should eq f32(40)
        (Tensor.from([1], [1], cuda) + Tensor.from([2], [1], cuda)).realize
        cuda.profile = nil
        profile.launches.should eq 3
        profile.kernels.keys.should eq ["Add [2,2]", "Mul [2,2]", "Sum []"]
        profile.host_ops.keys.should eq ["Add I32"]
        {profile.uploads, profile.downloads}.should eq({3, 1})
      end

      it "runs GPT-2's ops on the GPU, kernels and row ops included" do
        model = GPT2.load("#{__DIR__}/fixtures/tiny_gpt2/model.safetensors", heads: 4, device: cuda)
        host = cuda.host_ops
        model.forward([1, 2, 3], model.new_cache)
        (cuda.host_ops - host).should eq 0
      end
    end
  end

  describe "CUDA kernels compiled from PTX" do
    pending "no CUDA GPU on this machine" unless REAL_GPU

    if REAL_GPU
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
