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

#include <vector>

#include "encoder/enc_bit_writer.h"
#include "encoder/histogram.h"
#include "encoder/token.h"

namespace jxl {

// Data structure representing one element of the encoding table built
// from a distribution.
struct ANSEncSymbolInfo {
  uint16_t freq_;
  std::vector<uint16_t> reverse_map_;
  uint64_t ifreq_;
};

struct EntropyEncodingData {
  std::vector<std::vector<ANSEncSymbolInfo>> encoding_info;
};

void WriteHistograms(const std::vector<Histogram>& histograms,
                     EntropyEncodingData* codes, BitWriter* writer);

void WriteTokens(const std::vector<Token>& tokens,
                 const EntropyEncodingData& codes,
                 const std::vector<uint8_t>& context_map, BitWriter* writer);

}  // namespace jxl
#endif  // ENCODER_ENC_ANS_H_
