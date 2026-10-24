require "./spec_helper"

{% if flag?(:darwin) && flag?(:aarch64) %}
  describe Metal do
    pending "no Metal GPU on this machine" unless Metal.available?

    if Metal.available?
      metal = Metal.default

      it "names the GPU" do
        metal.name.should_not be_empty
      end

      it "runs ported ops on the GPU and the rest on the host" do
        gpu, host = metal.gpu_ops, metal.host_ops
        a = Tensor.from(f32(1, 2, 3, 4), [2, 2], metal)
        # a·a = [[7, 10], [15, 22]], which sums to 54.
        a.matmul(a).sum.to_a(Float32).should eq f32(54)
        metal.gpu_ops.should eq gpu + 2
        metal.host_ops.should eq host
        # I32 has no Metal kernels, so it runs on the host.
        (Tensor.from([1, 2], [2], metal) + Tensor.from([3, 4], [2], metal)).to_a(Int32).should eq [4, 6]
        metal.host_ops.should eq host + 1
      end

      it "mixes GPU and host ops in one graph" do
        a = Tensor.from(f32(1, 2, 3, 4), [2, 2], metal)
        ids = Tensor.from([1, 0], [2], metal) * Tensor.from([1, 1], [2], metal)
        # ids (I32, on the host) = [1, 0]; rows of a in that order = [[3, 4], [1, 2]];
        # · a = [[15, 22], [7, 10]]; summed by rows = [37, 17]
        a.gather(ids).matmul(a).sum(1).to_a(Float32).should eq f32(37, 17)
      end

      describe "reductions" do
        rng = Random.new(9)
        xs = Array(Float32).new(3 * 300) { rng.rand(-1f32...1f32) }
        sums = ->(data : Array(Float32), width : Int32) { Tensor.from(data, [3, width], metal).sum(1).to_a(Float32) }
        maxima = ->(data : Array(Float32), width : Int32) { Tensor.from(data, [3, width], metal).max(1).to_a(Float32) }
        # Each row of xs, followed by *count* copies of *value*.
        pad = ->(value : Float32, count : Int32) { xs.each_slice(300).to_a.flat_map { |row| row + Array.new(count, value) } }

        it "gives the same bits on every run" do
          first = sums.call(xs, 300)
          3.times { sums.call(xs, 300).should eq first }
        end

        it "keeps the bits of each sum when a row is padded with zeros" do
          # Masked attention rows differ between prefill and decode only by such padding.
          sums.call(pad.call(0f32, 211), 511).should eq sums.call(xs, 300)
        end

        it "keeps each maximum when a row is padded with -inf" do
          maxima.call(pad.call(-Float32::INFINITY, 211), 511).should eq maxima.call(xs, 300)
        end

        it "keeps each softmax value when a long row is padded with -inf" do
          # Masked attention rows differ between prefill and decode only by such padding.
          padded = pad.call(-Float32::INFINITY, 211)
          Tensor.from(padded, [3, 511], metal).softmax.slice(1, 0, 300).to_a(Float32)
            .should eq Tensor.from(xs, [3, 300], metal).softmax.to_a(Float32)
        end

        it "matches the host's maximum exactly" do
          maxima.call(xs, 300).should eq Tensor.from(xs, [3, 300]).max(1).to_a(Float32)
        end
      end

      describe "matmul modes" do
        rng = Random.new(5)
        xs = Array(Float32).new(3 * 200) { rng.rand(-1f32...1f32) }
        ws = Array(Float32).new(200 * 70) { rng.rand(-1f32...1f32) }
        product = ->(mode : Metal::MatMulMode) do
          metal.matmul_mode = mode
          Tensor.from(xs, [3, 200], metal).matmul(Tensor.from(ws, [200, 70], metal)).to_a(Float32)
      ensure
        metal.matmul_mode = Metal::MatMulMode::Auto
        end
        expected = Tensor.from(xs, [3, 200]).matmul(Tensor.from(ws, [200, 70])).to_a(Float32)

        it "is bit-identical to the host when serial" do
          product.call(Metal::MatMulMode::Serial).should eq expected
        end

        it "gives the same bits on every run when split" do
          first = product.call(Metal::MatMulMode::Split)
          3.times { product.call(Metal::MatMulMode::Split).should eq first }
        end

        it "keeps the bits of each output when k is padded with zeros" do
          # Prefill and decode differ only by masked positions, which add zeros
          # to k; chunks at fixed offsets make that change nothing.
          padded = xs.each_slice(200).to_a.flat_map { |row| row + Array.new(77, 0f32) }
          padded_ws = ws + Array.new(77 * 70, 0f32)
          split = product.call(Metal::MatMulMode::Split)
          Tensor.from(padded, [3, 277], metal).matmul(Tensor.from(padded_ws, [277, 70], metal)).to_a(Float32).should eq split
        end

        it "gives a row the same bits whatever the number of rows" do
          one_row = Tensor.from(xs[0, 200], [1, 200], metal).matmul(Tensor.from(ws, [200, 70], metal)).to_a(Float32)
          one_row.should eq product.call(Metal::MatMulMode::Split)[0, 70]
        end

        # Returns a·b under *mode* and *tile*; a is [*batch, m, k], and b is
        # [*batch, k, n], or [*batch, n, k] transposed when *b_transposed*.
        tiled_product = ->(mode : Metal::MatMulMode, tile : Metal::MatMulTile, a_shape : Array(Int32), b_shape : Array(Int32), b_transposed : Bool) do
          metal.matmul_mode, metal.matmul_tile = mode, tile
          rng = Random.new(a_shape.sum + b_shape.sum)
          a = Tensor.from(Array(Float32).new(a_shape.product) { rng.rand(-1f32...1f32) }, a_shape, metal)
          b = Tensor.from(Array(Float32).new(b_shape.product) { rng.rand(-1f32...1f32) }, b_shape, metal)
          a.matmul(b_transposed ? b.transpose : b).to_a(Float32)
      ensure
        metal.matmul_mode, metal.matmul_tile = Metal::MatMulMode::Auto, Metal::MatMulTile::T64x64
        end
        shapes = [
          {[3, 200], [200, 70], false}, {[70, 129], [129, 65], false}, {[64, 64], [64, 64], false},
          {[2, 37, 577], [2, 577, 33], false}, {[33, 100], [97, 100], true}, {[128, 768], [96, 768], true},
          {[5, 1], [1, 3], false},
        ]

        it "gives split's bits when tiled, with every tile, batched and transposed" do
          shapes.each do |(a_shape, b_shape, b_transposed)|
            split = tiled_product.call(Metal::MatMulMode::Split, Metal::MatMulTile::T64x64, a_shape, b_shape, b_transposed)
            Metal::MatMulTile.values.each do |tile|
              tiled_product.call(Metal::MatMulMode::Tiled, tile, a_shape, b_shape, b_transposed).should eq split
            end
          end
        end

        it "gives split's bits when automatic, whatever the number of rows" do
          [{[1, 300], [300, 50], false}, {[40, 300], [300, 50], false}].each do |(a_shape, b_shape, b_transposed)|
            tiled_product.call(Metal::MatMulMode::Auto, Metal::MatMulTile::T64x64, a_shape, b_shape, b_transposed)
              .should eq tiled_product.call(Metal::MatMulMode::Split, Metal::MatMulTile::T64x64, a_shape, b_shape, b_transposed)
          end
        end

        it "gives a tiled prefill the bits of a decode" do
          model = GPT2.load("#{__DIR__}/fixtures/tiny_gpt2/model.safetensors", heads: 4, device: metal)
          ids = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
          metal.matmul_mode = Metal::MatMulMode::Tiled
          prefill = model.forward(ids, model.new_cache).to_a(Float32)
          metal.matmul_mode = Metal::MatMulMode::Split
          cache = model.new_cache
          ids.flat_map { |id| model.forward([id], cache).to_a(Float32) }.should eq prefill
        ensure
          metal.matmul_mode = Metal::MatMulMode::Auto
        end

        it "stays close to the host when atomic" do
          product.call(Metal::MatMulMode::Atomic).zip(expected) { |a, e| a.should be_close(e, 1e-5 * Math.max(1f32, e.abs)) }
        end
      end

      it "profiles dispatches, host ops and GPU time per kernel" do
        a = Tensor.from(f32(1, 2, 3, 4), [2, 2], metal)
        profile = Metal::Profile.new(per_kernel: true)
        metal.profile = profile
        ((a + 1) * a).sum.realize
        (Tensor.from([1], [1], metal) + Tensor.from([2], [1], metal)).realize
        metal.profile = nil
        profile.dispatches.should eq 3
        profile.kernels.keys.sort.should eq ["Add", "Mul", "Sum"]
        profile.kernels.values.all? { |(count, seconds)| count == 1 && seconds >= 0 }.should be_true
        profile.host_ops.keys.should eq ["Add I32"]
        profile.command_buffers.should eq 3
      end

      it "commits a command buffer every commit_every dispatches" do
        a = Tensor.from(f32(1, 2, 3, 4), [2, 2], metal)
        metal.commit_every = 2
        profile = Metal::Profile.new
        metal.profile = profile
        (((a + 1) * a - a) / a).exp.to_a(Float32)
        metal.profile = nil
        metal.commit_every = 64
        profile.dispatches.should eq 5
        profile.command_buffers.should eq 3
      end

      it "reuses the Metal buffers of collected tensors" do
        # Collection is conservative and finalizers can run on another thread,
        # so memory comes back after some collection, not after a known one.
        hits = metal.buffer_pool.hits
        reused = (1..10).any? do
          20.times { (Tensor.from(f32(1, 2), [2], metal) * 2).to_a(Float32).should eq f32(2, 4) }
          GC.collect
          metal.buffer_pool.hits > hits
        end
        reused.should be_true
      end

      it "shares memory with the host" do
        t = Tensor.from(f32(1, 2), [2], metal)
        metal.map(t.node.buffer!).should_not be_nil
      end
    end
  end
{% end %}
