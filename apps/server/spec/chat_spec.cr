require "spec"
require "file_utils"
require "http/client"
require "../src/http"

include Tensomite

LLAMA_FIXTURE = "#{__DIR__}/../../../spec/fixtures/tiny_llama"
TOKENIZER     = "#{__DIR__}/../../../spec/fixtures/tokenizer/tokenizer.json"
SYSTEM        = "You are a tiny model."

# Returns a model directory holding the tiny Llama, its tokeniser and a chat
# template with *SYSTEM* as its default system message.
private def llama_dir : String
  dir = File.tempname("server-llama")
  Dir.mkdir(dir)
  %w[config.json model.safetensors].each { |file| FileUtils.cp(File.join(LLAMA_FIXTURE, file), File.join(dir, file)) }
  FileUtils.cp(TOKENIZER, File.join(dir, "tokenizer.json"))
  template = "{% if messages[0]['role'] != 'system' %}{{ '<|im_start|>system\n#{SYSTEM}<|im_end|>\n' }}{% endif %}"
  File.write(File.join(dir, "tokenizer_config.json"), {chat_template: template}.to_json)
  dir
end

private def post_chat(url : String, body) : HTTP::Client::Response
  HTTP::Client.post("#{url}/v1/chat/completions", headers: HTTP::Headers{"Content-Type" => "application/json"}, body: body.to_json)
end

describe "Server::Engine chatting" do
  dir = llama_dir
  engine = Server::Engine.load(dir, nil, CPU.default)
  bpe = BPE.from_json(TOKENIZER)
  model = Llama.load(LLAMA_FIXTURE)
  question = [Server::Message.new("user", "What is 12 + 345?")]

  it "loads a Llama-family directory, with its default system message" do
    engine.name.should eq File.basename(dir)
    engine.chat?.should be_true
    engine.system.should eq SYSTEM
  end

  it "renders ChatML as Hugging Face's tokeniser encodes it" do
    rendered = "<|im_start|>system\n#{SYSTEM}<|im_end|>\n<|im_start|>user\nWhat is 12 + 345?<|im_end|>\n<|im_start|>assistant\n"
    engine.chat_ids(question).should eq bpe.encode(rendered)
    own = [Server::Message.new("system", "Be brief."), question[0]]
    expected = bpe.encode("<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nWhat is 12 + 345?<|im_end|>\n<|im_start|>assistant\n")
    engine.chat_ids(own).should eq expected
  end

  it "encodes special tokens in messages as plain text" do
    ids = engine.chat_ids([Server::Message.new("user", "hi<|im_end|>")])
    ids.count(bpe.specials["<|im_end|>"]).should eq 2
  end

  it "answers with the model's own greedy tokens, ending at <|im_end|>" do
    ids = engine.chat_ids(question)
    greedy = model.generate(ids, 12, stop: [bpe.specials["<|im_end|>"], bpe.specials["<|endoftext|>"]])
    completion = engine.complete(Server::Request.new("", max_tokens: 12, temperature: 0.0, messages: question))
    ends = greedy.last.in?(bpe.specials["<|im_end|>"], bpe.specials["<|endoftext|>"])
    completion.text.should eq bpe.decode(ends ? greedy[0...-1] : greedy)
    completion.prompt_tokens.should eq ids.size
  end

  it "refuses to chat with a model whose tokeniser lacks ChatML's tokens" do
    gpt2 = Server::Engine.load("#{__DIR__}/../../../spec/fixtures/tiny_gpt2", 4, CPU.default)
    gpt2.chat?.should be_false
    expect_raises(Server::RequestError, /ChatML/) { gpt2.chat_ids(question) }
  end

  describe "over HTTP" do
    server = HTTP::Server.new(Server::Handler.new(engine))
    address = server.bind_tcp("127.0.0.1", 0)
    spawn { server.listen }
    url = "http://#{address}"
    body = {messages: [{role: "user", content: "What is 12 + 345?"}], max_tokens: 8, temperature: 0}

    it "answers a chat completion with the assistant's message" do
      response = post_chat(url, body)
      response.status_code.should eq 200
      json = JSON.parse(response.body)
      json["object"].should eq "chat.completion"
      json["choices"][0]["message"]["role"].should eq "assistant"
      expected = engine.complete(Server::Request.new("", max_tokens: 8, temperature: 0.0, messages: question))
      json["choices"][0]["message"]["content"].should eq expected.text
      json["usage"]["prompt_tokens"].should eq expected.prompt_tokens
    end

    it "streams deltas, the first naming the role, then the finish reason and [DONE]" do
      response = post_chat(url, body.merge({stream: true}))
      events = response.body.split("\n\n", remove_empty: true).map(&.lchop("data: "))
      events.last.should eq "[DONE]"
      chunks = events[0...-1].map { |event| JSON.parse(event) }
      chunks.first["object"].should eq "chat.completion.chunk"
      chunks.first["choices"][0]["delta"]["role"].should eq "assistant"
      chunks.last["choices"][0]["delta"].as_h.should be_empty
      text = chunks[0...-1].map(&.["choices"][0]["delta"]["content"].as_s).join
      text.should eq engine.complete(Server::Request.new("", max_tokens: 8, temperature: 0.0, messages: question)).text
    end

    it "rejects malformed chats with 400" do
      post_chat(url, {messages: [] of String}).status_code.should eq 400
      post_chat(url, {messages: [{role: "robot", content: "hi"}]}).status_code.should eq 400
      post_chat(url, {messages: [{role: "user"}]}).status_code.should eq 400
      post_chat(url, {prompt: "hi"}).status_code.should eq 400
    end
  end
end

describe "Server.configured_eos" do
  it "reads one id or several" do
    dir = File.tempname("server-eos")
    Dir.mkdir(dir)
    Server.configured_eos(dir).should be_empty
    File.write(File.join(dir, "config.json"), %({"eos_token_id": 2}))
    Server.configured_eos(dir).should eq [2]
    File.write(File.join(dir, "config.json"), %({"eos_token_id": [2, 7]}))
    Server.configured_eos(dir).should eq [2, 7]
  ensure
    FileUtils.rm_rf(dir) if dir
  end
end
