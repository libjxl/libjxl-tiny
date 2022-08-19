// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_ENC_ANS_PARAMS_H_
#define ENCODER_ENC_ANS_PARAMS_H_

// Encoder-only parameter needed for ANS entropy encoding methods.

#include <stdint.h>
#include <stdlib.h>

namespace jxl {

struct HistogramParams {
  enum class ClusteringType {
    kFastest,  // Only 4 clusters.
    kFast,
  };

  enum class LZ77Method {
    kNone,  // do not try lz77.
    kRLE,   // only try doing RLE.
  };

  HistogramParams() = default;

  ClusteringType clustering = ClusteringType::kFast;
  LZ77Method lz77_method = LZ77Method::kRLE;
  std::vector<size_t> image_widths;
  size_t max_histograms = ~0;
};

}  // namespace jxl

#endif  // ENCODER_ENC_ANS_PARAMS_H_
