#include "tokenizer.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "binary_io.h"

namespace cs336 {

namespace {

constexpr char kMagic[8] = {'C', 'S', '3', '3', '6', 'T', 'K', '\0'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kNoRank = std::numeric_limits<std::uint32_t>::max();

CodePointRanges read_ranges(BinaryReader& reader) {
  CodePointRanges result;
  const std::uint32_t count = reader.read_u32();
  result.ranges.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    const std::uint32_t lo = reader.read_u32();
    const std::uint32_t hi = reader.read_u32();
    result.ranges.emplace_back(lo, hi);
  }
  return result;
}

// Length of the valid UTF-8 sequence starting at s[i] (strict: no overlong forms, surrogates or
// code points above U+10FFFF), or 0 if the bytes are invalid. Stores the code point in `cp`.
std::size_t utf8_sequence(const std::string& s, std::size_t i, std::uint32_t& cp) {
  const auto byte = [&](std::size_t k) { return static_cast<unsigned char>(s[k]); };
  const unsigned char b0 = byte(i);
  const std::size_t remaining = s.size() - i;
  if (b0 < 0x80) {
    cp = b0;
    return 1;
  }
  std::size_t length;
  unsigned char low = 0x80, high = 0xBF;  // allowed range of the second byte
  if (b0 >= 0xC2 && b0 <= 0xDF) {
    length = 2;
    cp = b0 & 0x1F;
  } else if (b0 >= 0xE0 && b0 <= 0xEF) {
    length = 3;
    cp = b0 & 0x0F;
    if (b0 == 0xE0) low = 0xA0;
    if (b0 == 0xED) high = 0x9F;
  } else if (b0 >= 0xF0 && b0 <= 0xF4) {
    length = 4;
    cp = b0 & 0x07;
    if (b0 == 0xF0) low = 0x90;
    if (b0 == 0xF4) high = 0x8F;
  } else {
    return 0;
  }
  if (remaining < length) return 0;
  for (std::size_t k = 1; k < length; ++k) {
    const unsigned char b = byte(i + k);
    if (b < (k == 1 ? low : 0x80) || b > (k == 1 ? high : 0xBF)) return 0;
    cp = (cp << 6) | (b & 0x3F);
  }
  return length;
}

}  // namespace

bool CodePointRanges::contains(std::uint32_t cp) const {
  auto it = std::upper_bound(ranges.begin(), ranges.end(), std::make_pair(cp, std::numeric_limits<std::uint32_t>::max()));
  if (it == ranges.begin()) return false;
  --it;
  return cp >= it->first && cp <= it->second;
}

Tokenizer::Tokenizer(const std::string& path) {
  BinaryReader reader(path);
  char magic[8];
  reader.read_bytes(magic, sizeof magic);
  if (std::memcmp(magic, kMagic, sizeof magic) != 0) throw std::runtime_error(path + " is not a tokenizer.bin file");
  const std::uint32_t version = reader.read_u32();
  if (version != kVersion) throw std::runtime_error("unsupported tokenizer.bin version " + std::to_string(version));

  const std::uint32_t count = reader.read_u32();
  tokens_.reserve(count + 1);
  ranks_.reserve(count * 2);
  for (std::uint32_t i = 0; i < count; ++i) {
    tokens_.push_back(reader.read_string(reader.read_u32()));
    ranks_.emplace(tokens_.back(), i);
  }
  eos_id_ = static_cast<int>(reader.read_u32());
  eos_text_ = reader.read_string(reader.read_u32());
  if (eos_id_ != static_cast<int>(count)) throw std::runtime_error("unexpected end-of-text token id");
  tokens_.push_back(eos_text_);  // the special token is not a BPE rank: it is not added to ranks_
  letters_ = read_ranges(reader);
  numbers_ = read_ranges(reader);
  whitespace_ = read_ranges(reader);
}

const std::string& Tokenizer::token_bytes(int id) const {
  if (id < 0 || id >= num_tokens()) throw std::out_of_range("token id out of range");
  return tokens_[id];
}

std::string Tokenizer::decode(const std::vector<int>& tokens) const {
  std::string text;
  for (int id : tokens) text += token_bytes(id);
  return text;
}

std::vector<std::pair<std::size_t, std::size_t>> Tokenizer::pre_tokenize(const std::string& text) const {
  // Decode to code points, remembering where each one starts (offsets has one extra entry: the end).
  std::vector<std::uint32_t> cps;
  std::vector<std::size_t> offsets;
  for (std::size_t i = 0; i < text.size();) {
    std::uint32_t cp = 0;
    const std::size_t length = utf8_sequence(text, i, cp);
    if (length == 0) throw std::invalid_argument("input is not valid UTF-8 (byte offset " + std::to_string(i) + ")");
    cps.push_back(cp);
    offsets.push_back(i);
    i += length;
  }
  offsets.push_back(text.size());

  const std::size_t n = cps.size();
  const auto is_letter = [&](std::size_t i) { return letters_.contains(cps[i]); };
  const auto is_number = [&](std::size_t i) { return numbers_.contains(cps[i]); };
  const auto is_space = [&](std::size_t i) { return whitespace_.contains(cps[i]); };
  const auto is_other = [&](std::size_t i) { return !is_space(i) && !is_letter(i) && !is_number(i); };

  // Matches ` ?` followed by a possessive run of characters satisfying `pred`; returns the end or `start`.
  const auto spaced_run = [&](std::size_t start, const auto& pred) {
    std::size_t i = start;
    if (i < n && cps[i] == ' ') ++i;
    if (i >= n || !pred(i)) return start;
    while (i < n && pred(i)) ++i;
    return i;
  };

  std::vector<std::pair<std::size_t, std::size_t>> pieces;
  std::size_t i = 0;
  while (i < n) {
    std::size_t end = i;
    if (cps[i] == '\'' && i + 1 < n) {  // '(?:[sdmt]|ll|ve|re)
      const std::uint32_t c1 = cps[i + 1];
      const std::uint32_t c2 = i + 2 < n ? cps[i + 2] : 0;
      if (c1 == 's' || c1 == 'd' || c1 == 'm' || c1 == 't') {
        end = i + 2;
      } else if ((c1 == 'l' && c2 == 'l') || (c1 == 'v' && c2 == 'e') || (c1 == 'r' && c2 == 'e')) {
        end = i + 3;
      }
    }
    if (end == i) end = spaced_run(i, is_letter);
    if (end == i) end = spaced_run(i, is_number);
    if (end == i) end = spaced_run(i, is_other);
    if (end == i && is_space(i)) {
      std::size_t run_end = i;
      while (run_end < n && is_space(run_end)) ++run_end;
      if (run_end == n) {
        end = run_end;  // \s++$
      } else if (run_end - i > 1) {
        end = run_end - 1;  // \s+(?!\S): leave the last space for the following token
      } else {
        end = i + 1;  // \s
      }
    }
    pieces.emplace_back(offsets[i], offsets[end]);
    i = end;
  }
  return pieces;
}

void Tokenizer::encode_piece(const char* data, std::size_t size, std::vector<int>& out) const {
  const auto lookup = [&](std::size_t begin, std::size_t end) {
    const auto it = ranks_.find(std::string(data + begin, end - begin));
    return it == ranks_.end() ? kNoRank : it->second;
  };
  if (const std::uint32_t whole = lookup(0, size); whole != kNoRank) {
    out.push_back(static_cast<int>(whole));
    return;
  }

  // Byte pair merging as in tiktoken: parts[i].first is the start of part i, parts[i].second the
  // rank of merging parts i and i + 1 (the last entry only marks the end of the piece).
  std::vector<std::pair<std::size_t, std::uint32_t>> parts;
  parts.reserve(size + 1);
  for (std::size_t i = 0; i <= size; ++i) parts.emplace_back(i, kNoRank);
  const auto get_rank = [&](std::size_t index, std::size_t skip) {
    if (index + skip + 2 < parts.size()) return lookup(parts[index].first, parts[index + skip + 2].first);
    return kNoRank;
  };
  for (std::size_t i = 0; i + 2 < parts.size(); ++i) parts[i].second = get_rank(i, 0);
  while (parts.size() > 1) {
    std::uint32_t best_rank = kNoRank;
    std::size_t best = 0;
    for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
      if (parts[i].second < best_rank) {
        best_rank = parts[i].second;
        best = i;
      }
    }
    if (best_rank == kNoRank) break;
    parts[best].second = get_rank(best, 1);
    if (best > 0) parts[best - 1].second = get_rank(best - 1, 1);
    parts.erase(parts.begin() + best + 1);
  }
  for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
    out.push_back(static_cast<int>(lookup(parts[i].first, parts[i + 1].first)));
  }
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
  std::vector<int> ids;
  for (const auto& piece : pre_tokenize(text)) encode_piece(text.data() + piece.first, piece.second - piece.first, ids);
  return ids;
}

std::string Utf8StreamDecoder::push(const std::string& bytes) {
  pending_ += bytes;
  // Find the start of an incomplete trailing sequence (at most 3 bytes) and hold it back.
  std::size_t keep = 0;
  const std::size_t size = pending_.size();
  for (std::size_t back = 1; back <= std::min<std::size_t>(3, size); ++back) {
    const unsigned char b = static_cast<unsigned char>(pending_[size - back]);
    if ((b & 0xC0) == 0x80) continue;  // continuation byte: look further back for the lead byte
    const std::size_t needed = b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : b >= 0xC0 ? 2 : 1;
    if (needed > back) keep = back;
    break;
  }
  std::string ready = pending_.substr(0, size - keep);
  pending_.erase(0, size - keep);
  return ready;
}

std::string Utf8StreamDecoder::flush() {
  std::string rest;
  rest.swap(pending_);
  return rest;
}

}  // namespace cs336
