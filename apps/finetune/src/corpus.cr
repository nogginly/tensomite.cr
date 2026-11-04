require "../../../src/tensomite"

module Finetune
  # A tokenised corpus: a file of little-endian `UInt16` token ids, read through a
  # memory map, so a corpus of any size costs no memory until windows are read.
  # GPT-2's 50,257 tokens fit in 16 bits.
  #
  # ```
  # Finetune::Corpus.write("tokens.bin", bpe.encode(File.read("corpus.txt")))
  # corpus = Finetune::Corpus.new("tokens.bin")
  # corpus.window(1000, 257) # 257 token ids from position 1000
  # ```
  class Corpus
    # The number of tokens.
    getter size : Int64

    # Writes *ids* to *path*. Raises `ArgumentError` for an id that does not fit
    # in 16 bits.
    def self.write(path : String, ids : Array(Int32)) : Nil
      File.open(path, "wb") do |file|
        ids.each do |id|
          raise ArgumentError.new("token id #{id} does not fit in 16 bits") unless 0 <= id <= UInt16::MAX
          file.write_bytes(id.to_u16, IO::ByteFormat::LittleEndian)
        end
      end
    end

    def initialize(path : String)
      @file = Tensomite::MappedFile.new(path)
      @size = File.size(path).to_i64 // 2
    end

    # Returns *count* token ids from position *start*.
    def window(start : Int64, count : Int32) : Array(Int32)
      raise IndexError.new("window #{start}, #{count} of #{size} tokens") unless 0 <= start && start + count <= size
      bytes = @file.bytes(start * 2, count * 2)
      Array(Int32).new(count) { |i| IO::ByteFormat::LittleEndian.decode(UInt16, bytes[i * 2, 2]).to_i32 }
    end
  end
end
