# Serves a GPT-2 or Llama-family model over HTTP, runs one completion in the
# terminal, or chats there.
#
#     ops br server                       # builds bin/release/server
#     bin/release/server --model models/gpt2                              # the HTTP server
#     bin/release/server generate --model models/gpt2 "Once upon a time"  # one completion
#     bin/release/server chat --model models/smollm2-135m-instruct        # a chat
#
# A GPT-2 directory holds `model.safetensors`, `vocab.json` and `merges.txt`, as
# `scripts/fetch_gpt2.sh` downloads them or as tensomite saves a fine-tuned
# GPT-2; a Llama-family one holds `config.json`, `model.safetensors` and
# `tokenizer.json`, as `scripts/fetch_smollm2.sh` downloads them. See
# `apps/server/README.md` for the HTTP API. Run with `--help` for the options.
require "option_parser"
require "./src/http"
require "../../src/tensomite/blas"
{% if flag?(:darwin) && flag?(:aarch64) %}
  require "../../src/tensomite/metal"
{% end %}
{% if flag?(:cuda) %}
  require "../../src/tensomite/cuda"
{% end %}

devices = {"cpu" => -> { Tensomite::CPU.default.as(Tensomite::Device) }, "blas" => -> { Tensomite::BLAS.default.as(Tensomite::Device) }}
{% if flag?(:darwin) && flag?(:aarch64) %}
  devices["metal"] = -> { Tensomite::Metal.default.as(Tensomite::Device) }
{% end %}
{% if flag?(:cuda) %}
  devices["cuda"] = -> { Tensomite::CUDA.default.as(Tensomite::Device) }
{% end %}

model_dir, device_name = "models/gpt2", "blas"
heads = nil.as(Int32?)
host, port = "127.0.0.1", 8080
generate = chat = false
system = nil.as(String?)
prompt = nil.as(String?)
max_tokens, temperature, top_k, seed = 64, 1.0, 0, 0
stops = [] of String

OptionParser.parse do |parser|
  parser.banner = "Usage: server [options]\n       server generate [options] PROMPT"
  parser.on("--model DIR", "Directory of a GPT-2 or Llama-family model (#{model_dir})") { |dir| model_dir = dir }
  parser.on("--system TEXT", "System message for chats without one (the chat template's, if any)") { |value| system = value }
  parser.on("--heads N", "Attention heads (n_head in the model's config.json, or 12)") { |value| heads = value.to_i }
  parser.on("--device NAME", "#{devices.keys.join(", ")} (#{device_name})") { |name| device_name = name }
  parser.on("--host HOST", "Address to listen on (#{host})") { |value| host = value }
  parser.on("--port PORT", "Port to listen on (#{port})") { |value| port = value.to_i }
  sampling = -> do
    parser.on("--max-tokens N", "Tokens to write at most (#{max_tokens})") { |value| max_tokens = value.to_i }
    parser.on("--temperature X", "Sampling temperature, 0 for greedy (#{temperature})") { |value| temperature = value.to_f }
    parser.on("--top-k N", "Sample from the N likeliest tokens, 0 for all (#{top_k})") { |value| top_k = value.to_i }
    parser.on("--seed N", "Random seed (#{seed})") { |value| seed = value.to_i }
    parser.on("--stop TEXT", "End before TEXT; may be given up to four times") { |value| stops << value }
  end
  parser.on("generate", "Run one completion of PROMPT, streamed to the terminal, without a server") do
    generate = true
    parser.banner = "Usage: server generate [options] PROMPT"
    sampling.call
    parser.unknown_args { |args| prompt = args.join(" ") unless args.empty? }
  end
  parser.on("chat", "Chat in the terminal, a line per message, until end of input") do
    chat = true
    max_tokens = 512
    parser.banner = "Usage: server chat [options]"
    sampling.call
  end
  parser.on("-h", "--help", "Show this help") { puts parser; exit }
  parser.invalid_option { |flag| abort "unknown option #{flag}\n#{parser}" }
end

device = devices[device_name]?.try(&.call) || abort "unknown device #{device_name}; this build has #{devices.keys.join(", ")}"
engine = Server::Engine.load(model_dir, heads, device)
engine.system = system if system

if chat
  abort "#{engine.name}'s tokeniser has no ChatML tokens, so it cannot chat" unless engine.chat?
  messages = [] of Server::Message
  loop do
    print "> "
    break unless line = gets
    next if line.blank?
    messages << Server::Message.new("user", line)
    request = Server::Request.new("", max_tokens, temperature, top_k, stops, seed, messages)
    completion = begin
      engine.complete(request) { |piece| print piece; STDOUT.flush }
    rescue ex : Server::RequestError
      STDERR.puts ex.message
      messages.pop
      next
    end
    puts
    messages << Server::Message.new("assistant", completion.text)
  end
elsif generate
  text = prompt
  abort "generate needs a PROMPT" unless text
  print text
  start = Time.instant
  request = Server::Request.new(text, max_tokens, temperature, top_k, stops, seed)
  completion = begin
    engine.complete(request) { |piece| print piece; STDOUT.flush }
  rescue ex : Server::RequestError
    abort "\n#{ex.message}"
  end
  seconds = (Time.instant - start).total_seconds
  STDERR.puts "\n\n#{completion.completion_tokens} tokens after #{completion.prompt_tokens} in #{seconds.round(2)} s, " \
              "#{(completion.completion_tokens / seconds).round(1)} tokens/s; finished: #{completion.finish_reason}"
else
  server = HTTP::Server.new([HTTP::LogHandler.new, Server::Handler.new(engine)])
  address = server.bind_tcp(host, port)
  puts "Serving #{engine.name} on #{device_name} at http://#{address}"
  server.listen
end
