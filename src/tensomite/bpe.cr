require "json"

module Tensomite
  # GPT-2's byte-level byte-pair-encoding tokeniser.
  #
  # Text is split into words with GPT-2's pattern, each word's UTF-8 bytes are mapped
  # onto printable characters, and adjacent symbols are merged in order of merge
  # rank until no ranked pair remains. Any byte sequence round-trips.
  #
  # ```
  # bpe = BPE.from_files("vocab.json", "merges.txt")
  # ids = bpe.encode("Hello world") # => [15496, 995]
  # bpe.decode(ids)                 # => "Hello world"
  # ```
  #
  # From a Hugging Face `tokenizer.json`, it also knows the special tokens, such
  # as ChatML's `<|im_start|>`, each encoded whole wherever it appears in the
  # text, and the pre-tokeniser that splits every digit into a word of its own:
  #
  # ```
  # bpe = BPE.from_json("tokenizer.json")
  # bpe.encode("<|im_start|>user") # => [1, 4093]
  # ```
  class BPE
    # GPT-2's pre-tokeniser: contractions, letter runs, number runs, other
    # symbols, then whitespace, each optionally led by one space.
    PATTERN = Regex.new(
      %q('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+),
      Regex::Options::UCP)

    # Maps each byte to a printable character, the identity for printable Latin-1
    # and code points from 256 up for the rest, so tokens never hold raw control
    # or whitespace bytes.
    BYTE_TO_CHAR = begin
      printable = ('!'.ord..'~'.ord).to_a + ('¡'.ord..'¬'.ord).to_a + ('®'.ord..'ÿ'.ord).to_a
      extra = 0
      Array(Char).new(256) do |byte|
        if printable.includes?(byte)
          byte.chr
        else
          extra += 1
          (255 + extra).chr
        end
      end
    end

    CHAR_TO_BYTE = BYTE_TO_CHAR.each_with_index.to_h { |char, byte| {char, byte.to_u8} }

    @ids : Hash(String, Int32)
    @tokens : Hash(Int32, String)
    @ranks : Hash({String, String}, Int32)
    @cache = {} of String => Array(Int32)
    @specials : Hash(String, Int32)
    @special_pattern : Regex?
    @digits : Bool
    @prefix_space : Bool

    # Returns a tokeniser from a `vocab.json` (token to id) and a `merges.txt` (one
    # merge per line, best first, after a `#version` line).
    def self.from_files(vocab_path : String, merges_path : String) : BPE
      vocab = Hash(String, Int32).from_json(File.read(vocab_path))
      merges = File.read_lines(merges_path).reject { |line| line.starts_with?("#version") || line.blank? }
      new(vocab, merges.map { |line| a, b = line.split(' ', 2); {a, b} })
    end

    # Returns the tokeniser in a Hugging Face `tokenizer.json`: a byte-level BPE
    # model, its added tokens, and a pre-tokeniser of `ByteLevel`, alone or after
    # `Digits` with `individual_digits`.
    #
    # Raises `Error` for anything else, such as a normaliser, byte fallback or a
    # SentencePiece model, which this tokeniser would get wrong.
    def self.from_json(path : String) : BPE
      json = JSON.parse(File.read(path))
      model = json["model"]
      if model["type"].as_s != "BPE" || model["byte_fallback"]?.try(&.as_bool?) ||
         model["continuing_subword_prefix"]?.try(&.as_s?) || model["end_of_word_suffix"]?.try(&.as_s?)
        raise Error.new("#{path}: only byte-level BPE models are supported")
      end
      raise Error.new("#{path}: normalizers are not supported") unless json["normalizer"]?.try(&.raw).nil?
      digits, prefix_space = pre_tokenizer(json["pre_tokenizer"], path)
      vocab = model["vocab"].as_h.transform_values(&.as_i)
      merges = model["merges"].as_a.map do |merge|
        if pair = merge.as_a?
          {pair[0].as_s, pair[1].as_s}
        else
          a, b = merge.as_s.split(' ', 2)
          {a, b}
        end
      end
      specials = {} of String => Int32
      json["added_tokens"]?.try &.as_a.each do |token|
        if %w[lstrip rstrip single_word].any? { |flag| token[flag]?.try(&.as_bool) }
          raise Error.new("#{path}: added token #{token["content"]} strips or matches whole words, which is not supported")
        end
        specials[token["content"].as_s] = token["id"].as_i
      end
      new(vocab.merge(specials), merges, specials, digits: digits, prefix_space: prefix_space)
    end

    # Returns the tokeniser in a GGUF file's metadata: a `gpt2` (byte-level BPE)
    # model, its tokens and merges, its control and user-defined tokens as
    # special tokens, and a pre-tokeniser llama.cpp names `gpt2`, `default` or
    # `smollm` (which splits every digit off). Raises `Error` for others.
    def self.from_gguf(file : GGUF) : BPE
      meta = file.metadata
      model = meta["tokenizer.ggml.model"]?.try(&.as_s) || raise Error.new("#{file.path} holds no tokeniser")
      raise Error.new("#{file.path}: tokeniser model #{model} is not supported") unless model == "gpt2"
      pre = meta["tokenizer.ggml.pre"]?.try(&.as_s) || "default"
      raise Error.new("#{file.path}: pre-tokeniser #{pre} is not supported") unless pre.in?("gpt2", "default", "smollm")
      tokens = meta["tokenizer.ggml.tokens"].as_a.map(&.as_s)
      types = meta["tokenizer.ggml.token_type"]?.try(&.as_a.map(&.as_i)) || [] of Int32
      merges = meta["tokenizer.ggml.merges"].as_a.map do |merge|
        a, b = merge.as_s.split(' ', 2)
        {a, b}
      end
      ids = tokens.each_with_index.to_h { |token, id| {token, id} }
      specials = {} of String => Int32
      types.each_with_index { |type, id| specials[tokens[id]] = id if type.in?(3, 4) }
      new(ids, merges, specials, digits: pre == "smollm",
        prefix_space: meta["tokenizer.ggml.add_space_prefix"]?.try(&.as_bool) || false)
    end

    # Returns whether the pre-tokeniser *json* splits digits and adds a leading
    # space, raising for one this tokeniser does not implement.
    private def self.pre_tokenizer(json : JSON::Any, path : String) : {Bool, Bool}
      steps = json["type"].as_s == "Sequence" ? json["pretokenizers"].as_a : [json]
      digits = steps.size == 2 && steps[0]["type"].as_s == "Digits" && steps[0]["individual_digits"].as_bool
      level = steps.last
      unless (steps.size == 1 || digits) && level["type"].as_s == "ByteLevel" && level["use_regex"]?.try(&.as_bool) != false
        raise Error.new("#{path}: pre-tokenizer #{json.to_json} is not supported")
      end
      {digits, level["add_prefix_space"]?.try(&.as_bool) || false}
    end

    # Returns a tokeniser of the token ids *ids* and ranked *merges*. Text that
    # matches one of *specials* encodes to its id whole. With *digits*, each
    # digit is a word of its own; with *prefix_space*, a space is added before
    # text that does not start with one.
    def initialize(@ids : Hash(String, Int32), merges : Array({String, String}),
                   @specials : Hash(String, Int32) = {} of String => Int32,
                   @digits : Bool = false, @prefix_space : Bool = false)
      @tokens = @ids.invert
      @ranks = merges.each_with_index.to_h { |pair, rank| {pair, rank} }
      unless @specials.empty?
        @special_pattern = Regex.union(@specials.keys.sort_by! { |token| -token.size })
      end
    end

    # Returns the special tokens, by text.
    def specials : Hash(String, Int32)
      @specials.dup
    end

    # Returns the number of tokens in the vocabulary.
    def size : Int32
      @ids.size
    end

    # Returns the token ids of *text*. With *special* false, special tokens in
    # *text* are encoded as ordinary text, as untrusted input should be.
    def encode(text : String, special : Bool = true) : Array(Int32)
      pattern = @special_pattern
      return encode_text(text) unless special && pattern
      ids = [] of Int32
      start = 0
      text.scan(pattern) do |match|
        ids.concat(encode_text(text.byte_slice(start, match.byte_begin(0) - start))) if match.byte_begin(0) > start
        ids << @specials[match[0]]
        start = match.byte_end(0)
      end
      ids.concat(encode_text(text.byte_slice(start))) if start < text.bytesize
      ids
    end

    # Pre-tokenises *text*, which holds no special tokens, and encodes each word.
    #
    # 1. Adds a leading space, with `prefix_space`, unless there is one.
    # 2. Splits off each digit as a word of its own, with `digits`.
    # 3. Splits each piece with GPT-2's pattern, and merges each word.
    private def encode_text(text : String) : Array(Int32)
      text = " " + text if @prefix_space && !text.starts_with?(' ')
      pieces = @digits ? text.split(/(\p{N})/).reject(&.empty?) : [text]
      pieces.flat_map { |piece| piece.scan(PATTERN).flat_map { |match| encode_word(match[0]) } }
    end

    # Returns the text of *ids*. Byte sequences that are not valid UTF-8, such as a
    # character cut between tokens, become U+FFFD.
    def decode(ids : Enumerable(Int32)) : String
      String.new(decode_bytes(ids)).scrub
    end

    # Returns the bytes of *ids*, which may end partway through a character, as
    # when text is streamed a token at a time.
    def decode_bytes(ids : Enumerable(Int32)) : Bytes
      bytes = IO::Memory.new
      ids.each do |id|
        token = @tokens[id]? || raise Error.new("token id #{id} is not in the vocabulary")
        if @specials[token]? == id
          bytes << token
        else
          token.each_char { |char| bytes.write_byte(CHAR_TO_BYTE[char]) }
        end
      end
      bytes.to_slice
    end

    # Returns the id of *token*, such as GPT-2's `<|endoftext|>`, or `nil` when the
    # vocabulary lacks it.
    def id?(token : String) : Int32?
      @ids[token]?
    end

    private def encode_word(word : String) : Array(Int32)
      @cache[word] ||= merge(word.bytes.map { |byte| BYTE_TO_CHAR[byte].to_s }).map do |symbol|
        @ids[symbol]? || raise Error.new("symbol #{symbol.inspect} is not in the vocabulary")
      end
    end

    # Repeatedly merges the adjacent pair with the lowest rank, everywhere it
    # occurs left to right, until no adjacent pair has a rank.
    private def merge(symbols : Array(String)) : Array(String)
      while symbols.size > 1
        best = nil
        best_rank = Int32::MAX
        symbols.each_cons_pair do |a, b|
          if (rank = @ranks[{a, b}]?) && rank < best_rank
            best, best_rank = {a, b}, rank
          end
        end
        break unless best
        merged = [] of String
        i = 0
        while i < symbols.size
          if i < symbols.size - 1 && symbols[i] == best[0] && symbols[i + 1] == best[1]
            merged << best[0] + best[1]
            i += 2
          else
            merged << symbols[i]
            i += 1
          end
        end
        symbols = merged
      end
      symbols
    end
  end
end
