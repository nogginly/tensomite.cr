require "./spec_helper"
require "json"

private record TokenCase, text : String, ids : Array(Int32), decoded : String

private def token_cases(path : String) : Array(TokenCase)
  json = JSON.parse(File.read(path))
  texts = json["texts"].as_a.map(&.as_s)
  ids = json["ids"].as_a.map(&.as_a.map(&.as_i))
  decoded = json["decoded"].as_a.map(&.as_s)
  texts.each_index.map { |i| TokenCase.new(texts[i], ids[i], decoded[i]) }.to_a
end

private def write_json(json : String) : String
  File.tempname("tensomite", ".json").tap { |path| File.write(path, json) }
end

describe "BPE.from_json" do
  dir = "#{__DIR__}/fixtures/tokenizer"
  bpe = BPE.from_json("#{dir}/tokenizer.json")
  cases = token_cases("#{dir}/expected.json")

  it "encodes like Hugging Face's tokenizers, digits and special tokens included" do
    cases.each { |example| bpe.encode(example.text).should eq example.ids }
  end

  it "decodes special tokens as their text" do
    cases.each { |example| bpe.decode(bpe.encode(example.text)).should eq example.decoded }
  end

  it "finds special tokens, and encodes them as text when told to" do
    bpe.specials.keys.should eq ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    start = bpe.specials["<|im_start|>"]
    bpe.encode("<|im_start|>").should eq [start]
    plain = bpe.encode("<|im_start|>", special: false)
    plain.should_not contain start
    bpe.decode(plain).should eq "<|im_start|>"
  end

  it "rejects tokenisers it would get wrong" do
    json = JSON.parse(File.read("#{dir}/tokenizer.json")).as_h
    json["normalizer"] = JSON.parse(%({"type": "NFC"}))
    expect_raises(Error, /normalizers/) { BPE.from_json(write_json(json.to_json)) }
    json["normalizer"] = JSON::Any.new(nil)
    json["pre_tokenizer"] = JSON.parse(%({"type": "Whitespace"}))
    expect_raises(Error, /pre-tokenizer/) { BPE.from_json(write_json(json.to_json)) }
  end
end

describe "M9: SmolLM2's tokeniser" do
  dir = ENV["TENSOMITE_SMOLLM2_DIR"]? || ""
  if File.exists?("#{dir}/tokenizer.json")
    bpe = BPE.from_json("#{dir}/tokenizer.json")
    cases = token_cases("#{__DIR__}/fixtures/smollm2/expected.json")

    it "encodes the corpus and ChatML like Hugging Face's tokenizers" do
      cases.each { |example| bpe.encode(example.text).should eq example.ids }
    end

    it "decodes them back" do
      cases.each { |example| bpe.decode(bpe.encode(example.text)).should eq example.decoded }
    end
  else
    pending "needs TENSOMITE_SMOLLM2_DIR (see scripts/fetch_smollm2.sh)"
  end
end
