require "./spec_helper"
require "json"

BLOCK_FORMATS = {DType::Q8_0, DType::Q4_0, DType::Q4_1}

# Returns `x · wᵀ` in Float64 from the values each side's quantisation keeps,
# the weight dequantised and each activation row quantised to Q8_0 and back,
# and a bound on each output's difference from ggml's sum. That is rounding in
# `F32`, plus, for `Q4_1`, the `F16` rounding of each activation block's
# `d · Σq`, which ggml multiplies by the weight block's minimum.
private def reference(x : Array(Float32), rows : Int32, w : Tensor) : {Array(Float64), Array(Float64)}
  n, k = w.shape
  weights = w.dequantize.to_a(Float32)
  kept = Tensor.from(x, [rows, k]).quantize(DType::Q8_0).dequantize.to_a(Float32)
  sums = Array(Float64).new(rows * n, 0.0)
  bounds = Array(Float64).new(rows * n, 0.0)
  rows.times do |row|
    n.times do |col|
      total = (0...k).sum { |j| kept[row * k + j].to_f64 * weights[col * k + j] }
      bound = 1e-5 * Math.max(1.0, total.abs)
      if w.dtype.q4_1?
        (k // 32).times do |block|
          minimum = weights[col * k + block * 32, 32].min.abs
          bound += minimum * kept[row * k + block * 32, 32].sum.abs / 1024
        end
      end
      sums[row * n + col], bounds[row * n + col] = total, bound
    end
  end
  {sums, bounds}
end

# Returns the number of ops *device* has run on the host instead of its own
# kernels, or 0 for a device that runs everything on the host.
private def host_ops(device : Device) : Int32
  {% if flag?(:darwin) && flag?(:aarch64) %}
    return device.host_ops if device.is_a?(Metal)
  {% end %}
  {% if flag?(:cuda) %}
    return device.host_ops if device.is_a?(CUDA)
  {% end %}
  0
end

# Returns true for devices whose block kernels give the host's bits: the host
# devices, and CUDA, whose division is IEEE's.
private def bit_exact?(device : Device) : Bool
  {% if flag?(:cuda) %}
    return true if device.is_a?(CUDA)
  {% end %}
  DEVICES.includes?(device)
end

describe "Tensor#linear with block-format weights" do
  rows, n, k = 5, 7, 96
  x = random_values(rows * k, 1)
  w = random_values(n * k, 2)
  bias = random_values(n, 3)

  BLOCK_FORMATS.each do |dtype|
    it "sums integer block products as ggml does, for #{dtype}" do
      q = Tensor.from(w, [n, k]).quantize(dtype)
      got = Tensor.from(x, [rows, k]).linear(q).to_a(Float32)
      sums, bounds = reference(x, rows, q)
      got.each_with_index { |a, i| a.should be_close(sums[i], bounds[i]) }
    end

    (DEVICES + GPUS).each do |device|
      it "matches the CPU on #{device.class}, with a bias, for #{dtype}" do
        results = [CPU.default, device].map do |dev|
          q = Tensor.from(w, [n, k], dev).quantize(dtype)
          Tensor.from(x, [rows, k], dev).linear(q, Tensor.from(bias, [n], dev)).to_a(Float32)
        end
        # Metal's division may differ in the last bit, which can move one
        # activation by a step of its block's scale. CUDA's is IEEE's.
        if bit_exact?(device)
          results[1].should eq results[0]
        else
          results[1].zip(results[0]) { |a, e| a.should be_close(e, 1e-2 * Math.max(1f32, e.abs)) }
        end
      end
    end

    it "gives each row the same bits alone as in a batch, for #{dtype}" do
      q = Tensor.from(w, [n, k]).quantize(dtype)
      batched = Tensor.from(x, [rows, k]).linear(q).to_a(Float32)
      single = (0...rows).flat_map { |row| Tensor.from(x[row * k, k], [1, k]).linear(q).to_a(Float32) }
      single.should eq batched
    end
  end

  GPUS.each do |gpu|
    it "runs on #{gpu.class}'s own kernels, with no host op" do
      BLOCK_FORMATS.each do |dtype|
        q = Tensor.from(w, [n, k], gpu).quantize(dtype)
        input = Tensor.from(x, [rows, k], gpu)
        ids = Tensor.from([1, 0], [2], gpu)
        before = host_ops(gpu)
        Tensor.realize([input.linear(q, Tensor.from(bias, [n], gpu)), q.gather(ids)])
        host_ops(gpu).should eq before
      end
    end
  end

  it "reads rows of x through views, and a weight sliced by rows" do
    q = Tensor.from(w, [n, k]).quantize(DType::Q4_0)
    xt = Tensor.from(x, [rows, k])
    viewed = Tensor.from(x, [k, rows]).transpose
    expected = viewed.contiguous.linear(q).to_a(Float32)
    viewed.linear(q).to_a(Float32).should eq expected
    xt.linear(q.slice(0, 2, 5)).to_a(Float32).should eq xt.linear(q).to_a(Float32).each_slice(n).to_a.flat_map(&.[2...5])
  end

  it "is the F32 matmul with the transpose for an F32 weight" do
    weight = Tensor.from(w, [n, k])
    xt = Tensor.from(x, [rows, k])
    xt.linear(weight).to_a(Float32).should eq xt.matmul(weight.transpose).to_a(Float32)
  end

  it "rejects mismatched shapes and cannot be differentiated" do
    q = Tensor.from(w, [n, k]).quantize(DType::Q8_0)
    expect_raises(ShapeError) { Tensor.from(x[0, 64], [1, 64]).linear(q) }
    input = Tensor.from(x, [rows, k])
    expect_raises(Error, /quantised/) { Tensomite.grad(input.linear(q).sum, [input]) }
  end
end

describe "Gather from a block-format table" do
  table = random_values(6 * 64, 4)

  BLOCK_FORMATS.each do |dtype|
    (DEVICES + GPUS).each do |device|
      it "dequantises the rows it selects, for #{dtype} on #{device.class}" do
        q = Tensor.from(table, [6, 64], device).quantize(dtype)
        whole = q.dequantize.to_a(Float32)
        got = q.gather(Tensor.from([4, 0, 4], [3], device))
        got.dtype.should eq DType::F32
        got.to_a(Float32).should eq whole[256, 64] + whole[0, 64] + whole[256, 64]
        q.slice(0, 2, 6).gather(Tensor.from([2], [1], device)).to_a(Float32).should eq whole[256, 64]
      end
    end
  end
end

describe "Quantised models" do
  dir = "#{__DIR__}/fixtures/tiny_llama"
  json = JSON.parse(File.read("#{dir}/logits.json"))
  prompt = json["prompt_ids"].as_a.map(&.as_i)
  reference = json["logits"].as_a.map(&.as_a.map(&.as_f.to_f32))

  {"q8_0" => DType::Q8_0, "q4_0" => DType::Q4_0}.each do |name, dtype|
    it "load a #{dtype} GGUF as the safetensors model quantised by tensomite, to the bit" do
      from_gguf = Llama.load("#{dir}/model-#{name}.gguf")
      quantised = Llama.load(dir).quantize(dtype, embeddings: true)
      expected = quantised.forward(prompt, quantised.new_cache).to_a(Float32)
      from_gguf.forward(prompt, from_gguf.new_cache).to_a(Float32).should eq expected
    end
  end

  (DEVICES + GPUS).select(&.shape_invariant?).each do |device|
    it "give the same bits in prefill and in one-by-one decode on #{device.class}" do
      model = Llama.load("#{dir}/model-q4_0.gguf", device)
      prefill = model.forward(prompt, model.new_cache).to_a(Float32)
      cache = model.new_cache
      prompt.flat_map { |id| model.forward([id], cache).to_a(Float32) }.should eq prefill
    end
  end

  GPUS.each do |gpu|
    {"q8_0", "q4_0"}.each do |name|
      it "run #{name} on #{gpu.class} within 0.001 nats of the CPU's predictions" do
        cpu = Llama.load("#{dir}/model-#{name}.gguf")
        on_gpu = Llama.load("#{dir}/model-#{name}.gguf", gpu)
        log_p = cpu.forward(prompt, cpu.new_cache).log_softmax
        log_q = on_gpu.forward(prompt, on_gpu.new_cache).to(CPU.default).log_softmax
        divergence = (log_p.exp * (log_p - log_q)).sum(-1).mean.to_a(Float32)[0]
        divergence.should be < 0.001
      end
    end
  end

  it "stray less from the F32 logits at Q8_0 than at Q4_0" do
    width = 400
    errors = {"q8_0", "q4_0"}.map do |name|
      model = Llama.load("#{dir}/model-#{name}.gguf")
      logits = model.forward(prompt, model.new_cache).to_a(Float32)
      total = reference.each_with_index.sum { |row, i| row.each_with_index.sum { |value, j| (logits[i * width + j] - value).abs } }
      total / (reference.size * width)
    end
    errors[0].should be < errors[1] / 4
  end

  it "quantise GPT-2, its embeddings included" do
    gpt2 = "#{__DIR__}/fixtures/tiny_gpt2/model.safetensors"
    model = GPT2.load(gpt2, heads: 4).quantize(DType::Q8_0, embeddings: true)
    model.named_params["h.0.attn.c_attn.weight"].value.dtype.should eq DType::Q8_0
    model.named_params["wpe.weight"].value.dtype.should eq DType::Q8_0
    ids = [1, 2, 3, 4, 5]
    prefill = model.forward(ids, model.new_cache).to_a(Float32)
    cache = model.new_cache
    ids.flat_map { |id| model.forward([id], cache).to_a(Float32) }.should eq prefill
    plain = GPT2.load(gpt2, heads: 4)
    expect_close prefill, plain.forward(ids, plain.new_cache).to_a(Float32), 0.2
  end
end
