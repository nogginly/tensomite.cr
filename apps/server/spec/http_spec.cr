require "spec"
require "http/client"
require "../src/http"

include Tensomite

HTTP_FIXTURE = "#{__DIR__}/../../../spec/fixtures/tiny_gpt2"

# Starts a server for *engine* on a free local port and returns its base URL.
private def serve(engine : Server::Engine) : String
  server = HTTP::Server.new(Server::Handler.new(engine))
  address = server.bind_tcp("127.0.0.1", 0)
  spawn { server.listen }
  Fiber.yield
  "http://#{address}"
end

private def post(url : String, body) : HTTP::Client::Response
  HTTP::Client.post("#{url}/v1/completions", headers: HTTP::Headers{"Content-Type" => "application/json"}, body: body.to_json)
end

describe Server::Handler do
  engine = Server::Engine.load(HTTP_FIXTURE, heads: 4, device: CPU.default)
  url = serve(engine)
  prompt = "The quick brown fox"

  it "reports its health and model" do
    response = HTTP::Client.get("#{url}/health")
    response.status_code.should eq 200
    JSON.parse(response.body)["model"].should eq "tiny_gpt2"
  end

  it "answers a completion with its text and usage" do
    response = post(url, {prompt: prompt, max_tokens: 8, temperature: 0})
    response.status_code.should eq 200
    body = JSON.parse(response.body)
    expected = engine.complete(Server::Request.new(prompt, max_tokens: 8, temperature: 0.0))
    body["object"].should eq "text_completion"
    body["choices"][0]["text"].should eq expected.text
    body["choices"][0]["finish_reason"].should eq "length"
    body["usage"]["completion_tokens"].should eq 8
    body["usage"]["total_tokens"].should eq expected.prompt_tokens + 8
  end

  it "streams a completion as events, ending with its finish reason and [DONE]" do
    response = post(url, {prompt: prompt, max_tokens: 8, temperature: 0, stream: true})
    response.content_type.should eq "text/event-stream"
    events = response.body.split("\n\n", remove_empty: true).map(&.lchop("data: "))
    events.last.should eq "[DONE]"
    chunks = events[0...-1].map { |event| JSON.parse(event)["choices"][0] }
    chunks.last["finish_reason"].should eq "length"
    chunks.map(&.["text"].as_s).join.should eq engine.complete(Server::Request.new(prompt, max_tokens: 8, temperature: 0.0)).text
  end

  it "rejects malformed requests with 400" do
    post(url, {max_tokens: 8}).status_code.should eq 400
    post(url, {prompt: prompt, max_tokens: 0}).status_code.should eq 400
    post(url, {prompt: prompt, temperature: "hot"}).status_code.should eq 400
    post(url, {prompt: prompt, stop: [1]}).status_code.should eq 400
    post(url, {prompt: ""}).status_code.should eq 400
    response = HTTP::Client.post("#{url}/v1/completions", body: "not json")
    response.status_code.should eq 400
    JSON.parse(response.body)["error"]["message"].as_s.should_not be_empty
  end

  it "accepts stream: false and a single stop string" do
    response = post(url, {prompt: prompt, max_tokens: 4, stream: false, stop: "zzz"})
    response.status_code.should eq 200
  end

  it "answers 404 for an unknown path" do
    HTTP::Client.get("#{url}/v1/models/nope").status_code.should eq 404
  end
end
