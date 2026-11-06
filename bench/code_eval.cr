# Measures how well GPT-2 models write Crystal, on methods from the files a
# fine-tuning run on a code corpus held out for validation, so no model has
# trained on them.
#
#     scripts/make_code_corpus.sh                    # data/crystal/input.txt
#     bin/release/finetune --out runs/crystal --data data/crystal/input.txt --device cuda ...
#     crystal run --release -Dcuda bench/code_eval.cr -- --run runs/crystal --device cuda
#
# Each model is shown the file up to a method's `def` line, at most `--context`
# tokens of it, and is measured four ways:
#
# - body loss: the mean cross-entropy, in nats per token, of the method's real
#   body, the lower the better;
# - closes: its greedy completion ends with an `end` at the `def`'s indent within
#   `--max-tokens` tokens;
# - parses: the file, with the completion in place of the body, is valid Crystal,
#   as `crystal tool format` judges it; the compiler checks the types of only the
#   methods a program calls, so a type check of the file would pass any body;
# - first line and exact: the completion's first line, or all of it, is the
#   real one.
#
# The models default to `models/gpt2` and the run's `best/`; `--model DIR`, given
# once or more, replaces them.
require "json"
require "option_parser"
require "../src/tensomite/blas"
{% if flag?(:darwin) && flag?(:aarch64) %}
  require "../src/tensomite/metal"
{% end %}
{% if flag?(:cuda) %}
  require "../src/tensomite/cuda"
{% end %}

include Tensomite

module CodeEval
  # A method of a held-out file: the file's text up to its `def` line, without
  # the line's newline; its body from that newline to its `end`; and the rest.
  record Method, path : String, before : String, body : String, after : String, indent : String do
    def first_line : String
      body.lchop('\n').lines.first? || ""
    end
  end

  record Score, loss : Float64, tokens : Int32, closes : Bool, parses : Bool, first_line : Bool, exact : Bool, completion : String

  DEF  = /^(\s*)(?:(?:private|protected)\s+)?def\s+\S/
  FILE = /(?m)^# File: (.+)$/

  # Returns the text of the files that begin after the run's training tokens, as
  # `{path, text}`, from its `tokens.bin` and `split.json`.
  def self.held_out_files(run : String, bpe : BPE) : Array({String, String})
    bytes = File.read(File.join(run, "tokens.bin")).to_slice
    tokens = Array(Int32).new(bytes.size // 2) { |i| IO::ByteFormat::LittleEndian.decode(UInt16, bytes[i * 2, 2]).to_i32 }
    split = JSON.parse(File.read(File.join(run, "split.json")))["validation_starts"].as_i64.to_i
    training = bpe.decode_bytes(tokens[0, split]).size
    text = String.new(bpe.decode_bytes(tokens))
    headers = text.scan(FILE)
    headers.each_with_index.compact_map do |header, i|
      next if header.byte_begin(0) < training
      start = header.byte_end(0) + 2
      finish = i + 1 < headers.size ? headers[i + 1].byte_begin(0) - 1 : text.bytesize
      {header[1], text.byte_slice(start, finish - start)} if finish > start
    end.to_a
  end

  # Returns the methods of *text* whose `def` fits on one line and whose body,
  # free of macro code, runs from two to *lines* lines before its `end`.
  def self.methods(path : String, text : String, lines : Int32) : Array(Method)
    rows = text.lines(chomp: false)
    rows.each_with_index.compact_map do |row, i|
      match = row.match(DEF)
      next if match.nil? || row.includes?(';') || row.count('(') != row.count(')')
      indent = match[1]
      close = (i + 1...rows.size).find { |j| rows[j].chomp == "#{indent}end" }
      next unless close && 3 <= close - i <= lines + 1
      body = "\n" + rows[i + 1..close].join.chomp
      next if body.includes?("{%") || body.includes?("{{")
      Method.new(path, rows[0..i].join.chomp, body, "\n" + rows[close + 1..].join, indent)
    end.to_a
  end

  # Returns whether *text* is valid Crystal, by formatting a copy of it.
  def self.parses?(text : String) : Bool
    path = File.tempname("code_eval", ".cr")
    File.write(path, text)
    Process.run("crystal", ["tool", "format", path], output: Process::Redirect::Close, error: Process::Redirect::Close).success?
  ensure
    File.delete(path) if path && File.exists?(path)
  end

  # Returns *count* methods spread evenly through *candidates*, taking the next
  # when one's file does not parse as it is or its tokens do not split cleanly
  # between the prompt and the body.
  def self.pick(candidates : Array(Method), count : Int32, bpe : BPE) : Array(Method)
    picked = [] of Method
    count.times do |k|
      (k * candidates.size // count...candidates.size).each do |i|
        method = candidates[i]
        next if picked.includes?(method)
        whole = bpe.encode(method.before + method.body)
        prompt = bpe.encode(method.before)
        next unless whole[0, prompt.size] == prompt && parses?(method.before + method.body + method.after)
        picked << method
        break
      end
    end
    picked
  end

  # Returns the score of *model* on *method*.
  def self.score(model : GPT2, bpe : BPE, method : Method, context : Int32, max_tokens : Int32) : Score
    whole = bpe.encode(method.before + method.body)
    prompt_size = bpe.encode(method.before).size
    prompt = whole[Math.max(0, prompt_size - context)...prompt_size]
    body = whole[prompt_size..]
    loss = body_loss(model, prompt, body)
    text, closes = complete(model, bpe, prompt, method.indent, max_tokens)
    parses = closes && parses?(method.before + text + method.after)
    first = text.lchop('\n').lines.first? == method.first_line
    Score.new(loss, body.size, closes, parses, first, text == method.body, text)
  end

  # Returns the summed cross-entropy of *body* following *prompt*.
  def self.body_loss(model : GPT2, prompt : Array(Int32), body : Array(Int32)) : Float64
    logits = model.forward(prompt + body, model.new_cache)
    vocab = logits.shape[1]
    values = logits.to_a(Float32)
    body.each_with_index.sum do |token, j|
      row = values[(prompt.size - 1 + j) * vocab, vocab]
      top = row.max.to_f64
      top + Math.log(row.sum { |v| Math.exp(v - top) }) - row[token]
    end
  end

  # Returns the greedy completion after *prompt*, and whether it reached an `end`
  # at *indent* within *max_tokens* tokens, where it is cut.
  def self.complete(model : GPT2, bpe : BPE, prompt : Array(Int32), indent : String, max_tokens : Int32) : {String, Bool}
    close = /\n#{Regex.escape(indent)}end\n/
    stop = bpe.id?("<|endoftext|>")
    ids = [] of Int32
    model.generate(prompt, max_tokens) do |token|
      break if token == stop
      ids << token
      text = String.new(bpe.decode_bytes(ids))
      break if text.includes?("\n# File: ")
      if match = text.match(close)
        return {text[0, match.begin(0) + match[0].size - 1], true}
      end
    end
    {bpe.decode(ids), false}
  end
end

devices = {"cpu" => -> { CPU.default.as(Device) }, "blas" => -> { BLAS.default.as(Device) }}
{% if flag?(:darwin) && flag?(:aarch64) %}
  devices["metal"] = -> { Metal.default.as(Device) }
{% end %}
{% if flag?(:cuda) %}
  devices["cuda"] = -> { CUDA.default.as(Device) }
{% end %}

run, device_name, models = nil.as(String?), "blas", [] of String
count, context, max_tokens, verbose = 40, 320, 192, false
OptionParser.parse do |parser|
  parser.banner = "Usage: code_eval --run RUN [options]"
  parser.on("--run DIR", "A finetune run on a code corpus, whose held-out files to use") { |dir| run = dir }
  parser.on("--model DIR", "A model to measure; once or more (models/gpt2 and RUN/best)") { |dir| models << dir }
  parser.on("--device NAME", "#{devices.keys.join(", ")} (#{device_name})") { |name| device_name = name }
  parser.on("--methods N", "Methods to complete (#{count})") { |value| count = value.to_i }
  parser.on("--context N", "Tokens of the file before each method's body (#{context})") { |value| context = value.to_i }
  parser.on("--max-tokens N", "Tokens each completion may take, and the most a body may have (#{max_tokens})") { |value| max_tokens = value.to_i }
  parser.on("--verbose", "Print each method's body and each model's completion") { verbose = true }
  parser.on("-h", "--help", "Show this help") { puts parser; exit }
  parser.invalid_option { |flag| abort "unknown option #{flag}\n#{parser}" }
end
dir = run
abort "code_eval needs --run DIR" unless dir
device = devices[device_name]?.try(&.call) || abort "unknown device #{device_name}; this build has #{devices.keys.join(", ")}"
models = ["models/gpt2", File.join(dir, "best")] if models.empty?

bpe = BPE.from_files(File.join(models[0], "vocab.json"), File.join(models[0], "merges.txt"))
files = CodeEval.held_out_files(dir, bpe)
candidates = files.flat_map { |path, text| CodeEval.methods(path, text, max_tokens // 4) }
candidates.select! { |method| bpe.encode(method.body).size <= max_tokens }
methods = CodeEval.pick(candidates, count, bpe)
puts "#{methods.size} methods from #{files.size} held-out files (#{candidates.size} candidates), #{context} tokens of context"
puts

scores = models.map do |path|
  config = File.join(path, "config.json")
  heads = File.exists?(config) ? JSON.parse(File.read(config))["n_head"].as_i : 12
  model = GPT2.load(File.join(path, "model.safetensors"), heads, device)
  methods.map do |method|
    score = CodeEval.score(model, bpe, method, context, max_tokens)
    GC.collect
    score
  end
end

if verbose
  methods.each_with_index do |method, i|
    puts "## #{method.path}: #{method.before.lines.last.strip}"
    puts "real:#{method.body}"
    models.each_with_index do |path, which|
      score = scores[which][i]
      puts "#{path}#{score.closes ? "" : " (no end)"}:#{score.completion}"
    end
    puts
  end
end

puts "model|body loss|closes|parses|first line|exact"
puts "-----|---------|------|------|----------|-----"
models.each_with_index do |path, which|
  list = scores[which]
  loss = list.sum(&.loss) / Math.max(1, list.sum(&.tokens))
  shares = {list.count(&.closes), list.count(&.parses), list.count(&.first_line), list.count(&.exact)}.join('|') { |hits| "#{hits}/#{list.size}" }
  puts "#{path}|#{loss.round(3)}|#{shares}"
end
