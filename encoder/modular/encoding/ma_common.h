// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_MODULAR_ENCODING_MA_COMMON_H_
#define ENCODER_MODULAR_ENCODING_MA_COMMON_H_

#include <stddef.h>

namespace jxl {

enum MATreeContext : size_t {
  kSplitValContext = 0,
  kPropertyContext = 1,
  kPredictorContext = 2,
  kOffsetContext = 3,
  kMultiplierLogContext = 4,
  kMultiplierBitsContext = 5,

  kNumTreeContexts = 6,
};

static constexpr size_t kMaxTreeSize = 1 << 22;

}  // namespace jxl

#endif  // ENCODER_MODULAR_ENCODING_MA_COMMON_H_
