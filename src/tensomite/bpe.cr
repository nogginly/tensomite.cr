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

    # Returns a tokeniser from a `vocab.json` (token to id) and a `merges.txt` (one
    # merge per line, best first, after a `#version` line).
    def self.from_files(vocab_path : String, merges_path : String) : BPE
      vocab = Hash(String, Int32).from_json(File.read(vocab_path))
      merges = File.read_lines(merges_path).reject { |line| line.starts_with?("#version") || line.blank? }
      new(vocab, merges.map { |line| a, b = line.split(' ', 2); {a, b} })
    end

    def initialize(@ids : Hash(String, Int32), merges : Array({String, String}))
      @tokens = @ids.invert
      @ranks = merges.each_with_index.to_h { |pair, rank| {pair, rank} }
    end

    # Returns the number of tokens in the vocabulary.
    def size : Int32
      @ids.size
    end

    # Returns the token ids of *text*.
    def encode(text : String) : Array(Int32)
      text.scan(PATTERN).flat_map { |match| encode_word(match[0]) }
    end

    # Returns the text of *ids*. Byte sequences that are not valid UTF-8, such as a
    # character cut between tokens, become U+FFFD.
    def decode(ids : Enumerable(Int32)) : String
      bytes = IO::Memory.new
      ids.each do |id|
        token = @tokens[id]? || raise Error.new("token id #{id} is not in the vocabulary")
        token.each_char { |char| bytes.write_byte(CHAR_TO_BYTE[char]) }
      end
      String.new(bytes.to_slice).scrub
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
