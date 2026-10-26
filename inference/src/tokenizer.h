#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace cs336 {

// Sorted, inclusive code point ranges.
struct CodePointRanges {
  std::vector<std::pair<std::uint32_t, std::uint32_t>> ranges;
  bool contains(std::uint32_t cp) const;
};

// GPT-2 byte-level BPE tokenizer (same ids as tiktoken's "gpt2"), reading tokenizer.bin written by
// export_tokenizer.py. Only valid UTF-8 text is accepted by encode().
class Tokenizer {
 public:
  explicit Tokenizer(const std::string& path);

  // Throws std::invalid_argument if `text` is not valid UTF-8. Special tokens such as <|endoftext|>
  // are encoded as ordinary text.
  std::vector<int> encode(const std::string& text) const;

  // Concatenated bytes of the tokens (the end-of-text token yields its "<|endoftext|>" text).
  std::string decode(const std::vector<int>& tokens) const;
  const std::string& token_bytes(int id) const;

  int eos_id() const { return eos_id_; }
  int num_tokens() const { return static_cast<int>(tokens_.size()); }

  // Splits `text` into the pieces of the GPT-2 pre-tokenization regex
  // '(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s
  // and returns them as [begin, end) byte ranges.
  std::vector<std::pair<std::size_t, std::size_t>> pre_tokenize(const std::string& text) const;

 private:
  void encode_piece(const char* data, std::size_t size, std::vector<int>& out) const;

  std::vector<std::string> tokens_;  // indexed by id (== rank)
  std::string eos_text_;
  int eos_id_ = 0;
  std::unordered_map<std::string, std::uint32_t> ranks_;
  CodePointRanges letters_;
  CodePointRanges numbers_;
  CodePointRanges whitespace_;
};

// Converts a stream of token byte strings into text that can be printed incrementally: bytes of an
// incomplete trailing UTF-8 sequence are held back until the following token completes them.
class Utf8StreamDecoder {
 public:
  std::string push(const std::string& bytes);
  std::string flush();

 private:
  std::string pending_;
};

}  // namespace cs336
