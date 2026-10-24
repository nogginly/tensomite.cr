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
        metal.matmul_mode = Metal::MatMulMode::Split
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
        20.times { (Tensor.from(f32(1, 2), [2], metal) * 2).realize }
        5.times { GC.collect }
        hits = metal.buffer_pool.hits
        20.times { (Tensor.from(f32(1, 2), [2], metal) * 2).to_a(Float32).should eq f32(2, 4) }
        metal.buffer_pool.hits.should be > hits
      end

      it "shares memory with the host" do
        t = Tensor.from(f32(1, 2), [2], metal)
        metal.map(t.node.buffer!).should_not be_nil
      end
    end
  end
{% end %}
