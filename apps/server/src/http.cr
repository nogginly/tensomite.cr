require "http/server"
require "json"
require "./engine"

module Server
  # Serves an `Engine` over HTTP, in the shape of OpenAI's completions API:
  #
  # - `GET /health`: `{"status": "ok", "model": name}`;
  # - `POST /v1/completions`: a JSON request with `prompt` and, optionally,
  #   `max_tokens`, `temperature`, `top_k`, `stop` (a string or up to four),
  #   `seed` and `stream`. Without `stream` it answers with the whole completion;
  #   with it, with server-sent events, one per piece of text, then `[DONE]`;
  # - `POST /v1/chat/completions`: the same, with `messages` (each a `role` and
  #   `content`) in place of `prompt`, for a model that chats in ChatML.
  #
  # A malformed request gets status 400, and an unknown path 404, each with an
  # `{"error": {"message": …}}` body.
  class Handler
    include HTTP::Handler

    @count = 0

    def initialize(@engine : Engine)
    end

    def call(context : HTTP::Server::Context) : Nil
      request = context.request
      case {request.method, request.path}
      when {"GET", "/health"}
        json(context.response, 200) { |io| {status: "ok", model: @engine.name}.to_json(io) }
      when {"POST", "/v1/completions"}
        completions(context, chat: false)
      when {"POST", "/v1/chat/completions"}
        completions(context, chat: true)
      else
        error(context.response, 404, "no route for #{request.method} #{request.path}")
      end
    end

    private def completions(context : HTTP::Server::Context, chat : Bool) : Nil
      body = JSON.parse(context.request.body.try(&.gets_to_end) || "")
      request, stream = chat ? Server.parse_chat_request(body) : Server.parse_request(body)
      raise RequestError.new("this model's tokeniser has no ChatML tokens, so it cannot chat") if chat && !@engine.chat?
      @count += 1
      id, created = "#{chat ? "chatcmpl" : "cmpl"}-#{@count}", Time.utc.to_unix
      if stream
        stream(context.response, request, chat, id, created)
      else
        whole(context.response, request, chat, id, created)
      end
    rescue ex : JSON::ParseException | RequestError
      error(context.response, 400, ex.message || "bad request")
    end

    private def whole(response : HTTP::Server::Response, request : Request, chat : Bool, id : String, created : Int64) : Nil
      completion = @engine.complete(request)
      json(response, 200) do |io|
        choice = if chat
                   {index: 0, message: {role: "assistant", content: completion.text}, finish_reason: completion.finish_reason}
                 else
                   {text: completion.text, index: 0, finish_reason: completion.finish_reason}
                 end
        {
          id: id, object: chat ? "chat.completion" : "text_completion", created: created, model: @engine.name,
          choices: [choice],
          usage: {
            prompt_tokens:     completion.prompt_tokens,
            completion_tokens: completion.completion_tokens,
            total_tokens:      completion.prompt_tokens + completion.completion_tokens,
          },
        }.to_json(io)
      end
    end

    # Sends each piece of text as an event once it is ready, then one with the
    # finish reason, then `[DONE]`. A chat's pieces are deltas, the first naming
    # the assistant's role and the last empty. A client that disconnects ends
    # the completion at its next piece.
    private def stream(response : HTTP::Server::Response, request : Request, chat : Bool, id : String, created : Int64) : Nil
      response.status_code = 200
      response.content_type = "text/event-stream"
      response.headers["Cache-Control"] = "no-cache"
      first = true
      event = ->(text : String, reason : String?) do
        choice = if chat
                   delta = first ? {role: "assistant", content: text} : {content: text}
                   delta = {} of String => String if reason
                   {index: 0, delta: delta, finish_reason: reason}
                 else
                   {text: text, index: 0, finish_reason: reason}
                 end
        first = false
        chunk = {
          id: id, object: chat ? "chat.completion.chunk" : "text_completion", created: created, model: @engine.name,
          choices: [choice],
        }
        response << "data: " << chunk.to_json << "\n\n"
        response.flush
      end
      completion = @engine.complete(request) { |text| event.call(text, nil) }
      event.call("", completion.finish_reason)
      response << "data: [DONE]\n\n"
    rescue IO::Error
    end

    private def json(response : HTTP::Server::Response, status : Int32, & : IO ->) : Nil
      response.status_code = status
      response.content_type = "application/json"
      yield response
    end

    private def error(response : HTTP::Server::Response, status : Int32, message : String) : Nil
      json(response, status) { |io| {error: {message: message, type: "invalid_request_error"}}.to_json(io) }
    end
  end

  # Returns the `Request` a completions body asks for, and whether it asks to be
  # streamed. Raises `RequestError` for a missing prompt or a field of the wrong
  # type or range; unknown fields are ignored.
  def self.parse_request(body : JSON::Any) : {Request, Bool}
    fields = body.as_h? || raise RequestError.new("the body must be a JSON object")
    prompt = fields["prompt"]?.try(&.as_s?) || raise RequestError.new("`prompt` must be a string")
    options(fields, prompt, nil, 16)
  end

  # Returns the `Request` a chat body asks for, and whether it asks to be
  # streamed: `messages`, each with a `role` of `system`, `user` or `assistant`
  # and a string `content`, and the options of `parse_request`, with
  # `max_tokens` 512 unless given.
  def self.parse_chat_request(body : JSON::Any) : {Request, Bool}
    fields = body.as_h? || raise RequestError.new("the body must be a JSON object")
    list = fields["messages"]?.try(&.as_a?) || raise RequestError.new("`messages` must be an array")
    messages = list.map do |item|
      role = item.as_h?.try(&.["role"]?).try(&.as_s?)
      content = item.as_h?.try(&.["content"]?).try(&.as_s?)
      unless role.in?("system", "user", "assistant") && content
        raise RequestError.new("each message needs a `role` of system, user or assistant, and a string `content`")
      end
      Message.new(role.as(String), content)
    end
    raise RequestError.new("`messages` is empty") if messages.empty?
    options(fields, "", messages, 512)
  end

  private def self.options(fields : Hash(String, JSON::Any), prompt : String, messages : Array(Message)?, max_default : Int32) : {Request, Bool}
    max_tokens = integer(fields, "max_tokens", max_default)
    raise RequestError.new("`max_tokens` must be at least 1") if max_tokens < 1
    temperature = number(fields, "temperature", 1.0)
    raise RequestError.new("`temperature` must not be negative") if temperature < 0
    top_k = integer(fields, "top_k", 0)
    raise RequestError.new("`top_k` must not be negative") if top_k < 0
    stop = strings(fields, "stop")
    raise RequestError.new("`stop` may hold at most 4 strings") if stop.size > 4
    {Request.new(prompt, max_tokens, temperature, top_k, stop, integer(fields, "seed", 0), messages), boolean(fields, "stream", false)}
  end

  # Each returns field *name* of *fields*, or *default* when it is absent or
  # null, and raises `RequestError` when it has the wrong type.
  private def self.integer(fields : Hash(String, JSON::Any), name : String, default : Int32) : Int32
    case value = fields[name]?.try(&.raw)
    when Nil   then default
    when Int64 then Int32::MIN <= value <= Int32::MAX ? value.to_i32 : raise RequestError.new("`#{name}` is out of range")
    else            raise RequestError.new("`#{name}` must be an integer")
    end
  end

  # :ditto:
  private def self.number(fields : Hash(String, JSON::Any), name : String, default : Float64) : Float64
    case value = fields[name]?.try(&.raw)
    when Nil            then default
    when Int64, Float64 then value.to_f
    else                     raise RequestError.new("`#{name}` must be a number")
    end
  end

  # :ditto:
  private def self.boolean(fields : Hash(String, JSON::Any), name : String, default : Bool) : Bool
    case value = fields[name]?.try(&.raw)
    when Nil  then default
    when Bool then value
    else           raise RequestError.new("`#{name}` must be true or false")
    end
  end

  # Returns field *name*, a string or an array of strings, as an array: empty
  # when it is absent or null.
  private def self.strings(fields : Hash(String, JSON::Any), name : String) : Array(String)
    case value = fields[name]?.try(&.raw)
    when Nil    then [] of String
    when String then [value]
    when Array  then value.map { |item| item.as_s? || raise RequestError.new("`#{name}` must hold strings") }
    else             raise RequestError.new("`#{name}` must be a string or an array of strings")
    end
  end
end
