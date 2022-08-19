// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_ENC_ANS_H_
#define ENCODER_ENC_ANS_H_

// Library to encode the ANS population counts to the bit-stream and encode
// symbols based on the respective distributions.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

#include "encoder/ans_common.h"
#include "encoder/ans_params.h"
#include "encoder/base/bits.h"
#include "encoder/base/compiler_specific.h"
#include "encoder/base/status.h"
#include "encoder/dec_huffman.h"
#include "encoder/enc_ans_params.h"
#include "encoder/enc_bit_writer.h"
#include "encoder/fields.h"
#include "encoder/huffman_table.h"

namespace jxl {

#define USE_MULT_BY_RECIPROCAL

// precision must be equal to:  #bits(state_) + #bits(freq)
#define RECIPROCAL_PRECISION (32 + ANS_LOG_TAB_SIZE)

// Experiments show that best performance is typically achieved for a
// split-exponent of 3 or 4. Trend seems to be that '4' is better
// for large-ish pictures, and '3' better for rather small-ish pictures.
// This is plausible - the more special symbols we have, the better
// statistics we need to get a benefit out of them.

// Our hybrid-encoding scheme has dedicated tokens for the smallest
// (1 << split_exponents) numbers, and for the rest
// encodes (number of bits) + (msb_in_token sub-leading binary digits) +
// (lsb_in_token lowest binary digits) in the token, with the remaining bits
// then being encoded as data.
//
// Example with split_exponent = 4, msb_in_token = 2, lsb_in_token = 0.
//
// Numbers N in [0 .. 15]:
//   These get represented as (token=N, bits='').
// Numbers N >= 16:
//   If n is such that 2**n <= N < 2**(n+1),
//   and m = N - 2**n is the 'mantissa',
//   these get represented as:
// (token=split_token +
//        ((n - split_exponent) * 4) +
//        (m >> (n - msb_in_token)),
//  bits=m & (1 << (n - msb_in_token)) - 1)
// Specifically, we would get:
// N = 0 - 15:          (token=N, nbits=0, bits='')
// N = 16 (10000):      (token=16, nbits=2, bits='00')
// N = 17 (10001):      (token=16, nbits=2, bits='01')
// N = 20 (10100):      (token=17, nbits=2, bits='00')
// N = 24 (11000):      (token=18, nbits=2, bits='00')
// N = 28 (11100):      (token=19, nbits=2, bits='00')
// N = 32 (100000):     (token=20, nbits=3, bits='000')
// N = 65535:           (token=63, nbits=13, bits='1111111111111')
struct HybridUintConfig {
  uint32_t split_exponent;
  uint32_t split_token;
  uint32_t msb_in_token;
  uint32_t lsb_in_token;
  JXL_INLINE void Encode(uint32_t value, uint32_t* JXL_RESTRICT token,
                         uint32_t* JXL_RESTRICT nbits,
                         uint32_t* JXL_RESTRICT bits) const {
    if (value < split_token) {
      *token = value;
      *nbits = 0;
      *bits = 0;
    } else {
      uint32_t n = FloorLog2Nonzero(value);
      uint32_t m = value - (1 << n);
      *token = split_token +
               ((n - split_exponent) << (msb_in_token + lsb_in_token)) +
               ((m >> (n - msb_in_token)) << lsb_in_token) +
               (m & ((1 << lsb_in_token) - 1));
      *nbits = n - msb_in_token - lsb_in_token;
      *bits = (value >> lsb_in_token) & ((1UL << *nbits) - 1);
    }
  }

  explicit HybridUintConfig(uint32_t split_exponent = 4,
                            uint32_t msb_in_token = 2,
                            uint32_t lsb_in_token = 0)
      : split_exponent(split_exponent),
        split_token(1 << split_exponent),
        msb_in_token(msb_in_token),
        lsb_in_token(lsb_in_token) {
    JXL_DASSERT(split_exponent >= msb_in_token + lsb_in_token);
  }
};

// Data structure representing one element of the encoding table built
// from a distribution.
// TODO(veluca): split this up, or use an union.
struct ANSEncSymbolInfo {
  // ANS
  uint16_t freq_;
  std::vector<uint16_t> reverse_map_;
#ifdef USE_MULT_BY_RECIPROCAL
  uint64_t ifreq_;
#endif
  // Prefix coding.
  uint8_t depth;
  uint16_t bits;
};

class ANSCoder {
 public:
  ANSCoder() : state_(ANS_SIGNATURE << 16) {}

  uint32_t PutSymbol(const ANSEncSymbolInfo& t, uint8_t* nbits) {
    uint32_t bits = 0;
    *nbits = 0;
    if ((state_ >> (32 - ANS_LOG_TAB_SIZE)) >= t.freq_) {
      bits = state_ & 0xffff;
      state_ >>= 16;
      *nbits = 16;
    }
#ifdef USE_MULT_BY_RECIPROCAL
    // We use mult-by-reciprocal trick, but that requires 64b calc.
    const uint32_t v = (state_ * t.ifreq_) >> RECIPROCAL_PRECISION;
    const uint32_t offset = t.reverse_map_[state_ - v * t.freq_];
    state_ = (v << ANS_LOG_TAB_SIZE) + offset;
#else
    state_ = ((state_ / t.freq_) << ANS_LOG_TAB_SIZE) +
             t.reverse_map_[state_ % t.freq_];
#endif
    return bits;
  }

  uint32_t GetState() const { return state_; }

 private:
  uint32_t state_;
};

struct LZ77Params : public Fields {
  LZ77Params() { Bundle::Init(this); }
  JXL_FIELDS_NAME(LZ77Params)
  Status VisitFields(Visitor* JXL_RESTRICT visitor) override {
    JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &enabled));
    if (!visitor->Conditional(enabled)) return true;
    JXL_QUIET_RETURN_IF_ERROR(visitor->U32(
        Val(224), Val(512), Val(4096), BitsOffset(15, 8), 224, &min_symbol));
    JXL_QUIET_RETURN_IF_ERROR(visitor->U32(Val(3), Val(4), BitsOffset(2, 5),
                                           BitsOffset(8, 9), 3, &min_length));
    return true;
  }
  bool enabled;

  // Symbols above min_symbol use a special hybrid uint encoding and
  // represent a length, to be added to min_length.
  uint32_t min_symbol;
  uint32_t min_length;

  // Not serialized by VisitFields.
  HybridUintConfig length_uint_config{0, 0, 0};

  size_t nonserialized_distance_context;
};

struct ANSCode {
  CacheAlignedUniquePtr alias_tables;
  std::vector<HuffmanDecodingData> huffman_data;
  std::vector<HybridUintConfig> uint_config;
  std::vector<int> degenerate_symbols;
  bool use_prefix_code;
  uint8_t log_alpha_size;  // for ANS.
  LZ77Params lz77;
  // Maximum number of bits necessary to represent the result of a
  // ReadHybridUint call done with this ANSCode.
  size_t max_num_bits = 0;
  void UpdateMaxNumBits(size_t ctx, size_t symbol);
};

// RebalanceHistogram requires a signed type.
using ANSHistBin = int32_t;

struct EntropyEncodingData {
  std::vector<std::vector<ANSEncSymbolInfo>> encoding_info;
  std::vector<HybridUintConfig> uint_config;
  LZ77Params lz77;
};

// Integer to be encoded by an entropy coder, either ANS or Huffman.
struct Token {
  Token() {}
  Token(uint32_t c, uint32_t value)
      : is_lz77_length(false), context(c), value(value) {}
  uint32_t is_lz77_length : 1;
  uint32_t context : 31;
  uint32_t value;
};

// Apply context clustering, compute histograms and encode them. Returns an
// estimate of the total bits used for encoding the stream. If `writer` ==
// nullptr, the bit estimate will not take into account the context map (which
// does not get written if `num_contexts` == 1).
void BuildAndEncodeHistograms(const HistogramParams& params,
                              size_t num_contexts,
                              std::vector<std::vector<Token>>& tokens,
                              EntropyEncodingData* codes,
                              std::vector<uint8_t>* context_map,
                              BitWriter* writer);

// Write the tokens to a string.
void WriteTokens(const std::vector<Token>& tokens,
                 const EntropyEncodingData& codes,
                 const std::vector<uint8_t>& context_map, BitWriter* writer);

// Exposed for tests; to be used with Writer=BitWriter only.
template <typename Writer>
void EncodeUintConfigs(const std::vector<HybridUintConfig>& uint_config,
                       Writer* writer, size_t log_alpha_size);
extern template void EncodeUintConfigs(const std::vector<HybridUintConfig>&,
                                       BitWriter*, size_t);

}  // namespace jxl
#endif  // ENCODER_ENC_ANS_H_
