// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef LIB_JXL_ENC_ANS_PARAMS_H_
#define LIB_JXL_ENC_ANS_PARAMS_H_

// Encoder-only parameter needed for ANS entropy encoding methods.

#include <stdint.h>
#include <stdlib.h>

#include "lib/jxl/enc_params.h"

namespace jxl {

struct HistogramParams {
  enum class ClusteringType {
    kFastest,  // Only 4 clusters.
    kFast,
  };

  enum class HybridUintMethod {
    kNone,        // just use kHybridUint420Config.
    k000,         // force the fastest option.
    kFast,        // just try a couple of options.
    kContextMap,  // fast choice for ctx map.
  };

  enum class LZ77Method {
    kNone,     // do not try lz77.
    kRLE,      // only try doing RLE.
    kLZ77,     // try lz77 with backward references.
    kOptimal,  // optimal-matching LZ77 parsing.
  };

  enum class ANSHistogramStrategy {
    kFast,         // Only try some methods, early exit.
    kApproximate,  // Only try some methods.
    kPrecise,      // Try all methods.
  };

  HistogramParams() = default;

  HistogramParams(SpeedTier tier, size_t num_ctx) {
    if (tier > SpeedTier::kFalcon) {
      clustering = ClusteringType::kFastest;
      lz77_method = LZ77Method::kNone;
    }
  }

  ClusteringType clustering = ClusteringType::kFast;
  HybridUintMethod uint_method = HybridUintMethod::kNone;
  LZ77Method lz77_method = LZ77Method::kRLE;
  ANSHistogramStrategy ans_histogram_strategy =
      ANSHistogramStrategy::kApproximate;
  std::vector<size_t> image_widths;
  size_t max_histograms = ~0;
  bool force_huffman = false;
};

}  // namespace jxl

#endif  // LIB_JXL_ENC_ANS_PARAMS_H_
