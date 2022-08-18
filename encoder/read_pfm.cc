// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/read_pfm.h"

#include "tools/file_io.h"

namespace jxl {

namespace {
class Parser {
 public:
  explicit Parser(const uint8_t* data, const size_t len)
      : pos_(data), end_(data + len) {}

  // Sets "pos" to the first non-header byte/pixel on success.
  bool ParseHeaderPFM(const uint8_t** pos, size_t* xsize, size_t* ysize,
                      bool* big_endian) {
    if (pos_[0] != 'P' || pos_[1] != 'F') return false;
    pos_ += 2;

    double scale;
    if (!SkipSingleWhitespace() || !ParseUnsigned(xsize) || !SkipBlank() ||
        !ParseUnsigned(ysize) || !SkipSingleWhitespace() ||
        !ParseSigned(&scale) || !SkipSingleWhitespace()) {
      return false;
    }

    // The scale has no meaning as multiplier, only its sign is used to
    // indicate endianness. All software expects nominal range 0..1.
    if (scale == 0.0 || std::abs(scale) != 1.0) {
      fprintf(stderr, "PFM: bad scale factor value.\n");
      return false;
    }
    *big_endian = scale > 0.0;

    *pos = pos_;
    return true;
  }

  bool ParseUnsigned(size_t* number) {
    if (pos_ == end_) {
      fprintf(stderr, "PNM: reached end before number.\n");
      return false;
    }
    if (!IsDigit(*pos_)) {
      fprintf(stderr, "PNM: expected unsigned number.\n");
      return false;
    }
    *number = 0;
    while (pos_ < end_ && *pos_ >= '0' && *pos_ <= '9') {
      *number *= 10;
      *number += *pos_ - '0';
      ++pos_;
    }

    return true;
  }

  bool ParseSigned(double* number) {
    if (pos_ == end_) {
      fprintf(stderr, "PNM: reached end before signed.\n");
      return false;
    }
    if (*pos_ != '-' && *pos_ != '+' && !IsDigit(*pos_)) {
      fprintf(stderr, "PNM: expected signed number.\n");
      return false;
    }

    // Skip sign
    const bool is_neg = *pos_ == '-';
    if (is_neg || *pos_ == '+') {
      ++pos_;
      if (pos_ == end_) {
        fprintf(stderr, "PNM: reached end before digits.\n");
        return false;
      }
    }

    // Leading digits
    *number = 0.0;
    while (pos_ < end_ && *pos_ >= '0' && *pos_ <= '9') {
      *number *= 10;
      *number += *pos_ - '0';
      ++pos_;
    }

    // Decimal places?
    if (pos_ < end_ && *pos_ == '.') {
      ++pos_;
      double place = 0.1;
      while (pos_ < end_ && *pos_ >= '0' && *pos_ <= '9') {
        *number += (*pos_ - '0') * place;
        place *= 0.1;
        ++pos_;
      }
    }

    if (is_neg) *number = -*number;
    return true;
  }

 private:
  static bool IsDigit(const uint8_t c) { return '0' <= c && c <= '9'; }
  static bool IsLineBreak(const uint8_t c) { return c == '\r' || c == '\n'; }
  static bool IsWhitespace(const uint8_t c) {
    return IsLineBreak(c) || c == '\t' || c == ' ';
  }

  bool SkipBlank() {
    if (pos_ == end_) {
      fprintf(stderr, "PNM: reached end before blank.\n");
      return false;
    }
    const uint8_t c = *pos_;
    if (c != ' ' && c != '\n') {
      fprintf(stderr, "PNM: expected blank.\n");
      return false;
    }
    ++pos_;
    return true;
  }

  bool SkipSingleWhitespace() {
    if (pos_ == end_) {
      fprintf(stderr, "PNM: reached end before whitespace.\n");
      return false;
    }
    if (!IsWhitespace(*pos_)) {
      fprintf(stderr, "PNM: expected whitespace.\n");
      return false;
    }
    ++pos_;
    return true;
  }

  const uint8_t* pos_;
  const uint8_t* const end_;
};

#if JXL_COMPILER_MSVC
#define JXL_BSWAP32(x) _byteswap_ulong(x)
#else
#define JXL_BSWAP32(x) __builtin_bswap32(x)
#endif

inline float BSwapFloat(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  uint32_t uswap = JXL_BSWAP32(u);
  float xswap;
  memcpy(&xswap, &uswap, 4);
  return xswap;
}

}  // namespace

bool ReadPFM(const char* fn, Image3F* image) {
  std::vector<uint8_t> data;
  if (!jpegxl::tools::ReadFile(fn, &data)) {
    fprintf(stderr, "Could not read %s\n", fn);
    return false;
  }
  if (data.size() < 2) {
    fprintf(stderr, "PFM file too small.\n");
    return false;
  }

  Parser parser(data.data(), data.size());
  size_t xsize, ysize;
  bool big_endian;
  const uint8_t* pos = nullptr;
  if (!parser.ParseHeaderPFM(&pos, &xsize, &ysize, &big_endian)) {
    return false;
  }

  Image3F img(xsize, ysize);
  const float* input = reinterpret_cast<const float*>(pos);
  const size_t stride = xsize * 3;
  for (size_t y = 0; y < ysize; ++y) {
    size_t y_in = ysize - 1 - y;
    const float* row_in = &input[y_in * stride];
    for (size_t c = 0; c < 3; ++c) {
      float* row_out = img.PlaneRow(c, y);
      for (size_t x = 0; x < xsize; ++x) {
        row_out[x] = row_in[x * 3 + c];
        if (big_endian) row_out[x] = BSwapFloat(row_out[x]);
      }
    }
  }

  *image = std::move(img);
  return true;
}

}  // namespace jxl
