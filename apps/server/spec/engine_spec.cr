require "spec"
require "../src/engine"

include Tensomite

FIXTURE = "#{__DIR__}/../../../spec/fixtures/tiny_gpt2"

# The tiny model's tokeniser, with `<|endoftext|>` added as *id* when given.
private def tokenizer(end_of_text : Int32? = nil) : BPE
  vocab = Hash(String, Int32).from_json(File.read("#{FIXTURE}/vocab.json"))
  vocab["<|endoftext|>"] = end_of_text if end_of_text
  merges = File.read_lines("#{FIXTURE}/merges.txt").reject { |line| line.starts_with?("#version") || line.blank? }
  BPE.new(vocab, merges.map { |line| a, b = line.split(' ', 2); {a, b} })
end

describe Server::Engine do
  model = GPT2.load("#{FIXTURE}/model.safetensors", heads: 4)
  bpe = tokenizer
  engine = Server::Engine.new(model, bpe, "tiny")
  prompt = "The quick brown fox"
  greedy = model.generate(bpe.encode(prompt), 12)

  it "writes the model's own greedy tokens, as text" do
    completion = engine.complete(Server::Request.new(prompt, max_tokens: 12, temperature: 0.0))
    completion.text.should eq bpe.decode(greedy)
    completion.prompt_tokens.should eq bpe.encode(prompt).size
    completion.completion_tokens.should eq 12
    completion.finish_reason.should eq "length"
  end

  it "yields pieces that join into the completion's text" do
    pieces = [] of String
    completion = engine.complete(Server::Request.new(prompt, max_tokens: 12, temperature: 0.8, seed: 3)) { |text| pieces << text }
    pieces.join.should eq completion.text
    pieces.should_not contain ""
  end

  it "repeats a sampled completion with the same seed" do
    request = Server::Request.new(prompt, max_tokens: 10, temperature: 1.0, top_k: 20, seed: 7)
    engine.complete(request).text.should eq engine.complete(request).text
  end

  it "ends before a stop string, never yielding any of it" do
    text = bpe.decode(greedy)
    stop = text[3, 2]
    pieces = [] of String
    completion = engine.complete(Server::Request.new(prompt, max_tokens: 12, temperature: 0.0, stop: [stop])) { |piece| pieces << piece }
    completion.text.should eq text[0, text.index!(stop)]
    completion.finish_reason.should eq "stop"
    pieces.join.should eq completion.text
  end

  it "ends at the end-of-text token, without writing it" do
    ended = Server::Engine.new(model, tokenizer(end_of_text: greedy[2]), "tiny")
    completion = ended.complete(Server::Request.new(prompt, max_tokens: 12, temperature: 0.0))
    completion.completion_tokens.should eq 2
    completion.finish_reason.should eq "stop"
  end

  it "stops at the end of the context" do
    long = (["the"] * 60).join(" ")
    completion = engine.complete(Server::Request.new(long, max_tokens: 100, temperature: 0.0))
    (completion.prompt_tokens + completion.completion_tokens).should eq model.config.context
    completion.finish_reason.should eq "length"
  end

  it "rejects an empty prompt and one that fills the context" do
    expect_raises(Server::RequestError, /empty/) { engine.complete(Server::Request.new("")) }
    expect_raises(Server::RequestError, /no room/) { engine.complete(Server::Request.new((["the"] * 100).join(" "))) }
  end
end

describe "Server.whole_characters" do
  it "drops a character cut at the end, and keeps whole ones" do
    bytes = "café".to_slice
    Server.whole_characters(bytes[0, bytes.size - 1]).should eq "caf"
    Server.whole_characters(bytes).should eq "café"
  end
end
