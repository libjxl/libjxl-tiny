// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_ENTROPY_CODER_H_
#define ENCODER_ENTROPY_CODER_H_

#include <stddef.h>
#include <stdint.h>

#include "encoder/ac_context.h"
#include "encoder/base/compiler_specific.h"
#include "encoder/field_encodings.h"

// Entropy coding and context modeling of DC and AC coefficients, as well as AC
// strategy and quantization field.

namespace jxl {

static JXL_INLINE int32_t PredictFromTopAndLeft(
    const int32_t* const JXL_RESTRICT row_top,
    const int32_t* const JXL_RESTRICT row, size_t x, int32_t default_val) {
  if (x == 0) {
    return row_top == nullptr ? default_val : row_top[x];
  }
  if (row_top == nullptr) {
    return row[x - 1];
  }
  return (row_top[x] + row[x - 1] + 1) / 2;
}

static constexpr U32Enc kDCThresholdDist(Bits(4), BitsOffset(8, 16),
                                         BitsOffset(16, 272),
                                         BitsOffset(32, 65808));

static constexpr U32Enc kQFThresholdDist(Bits(2), BitsOffset(3, 4),
                                         BitsOffset(5, 12), BitsOffset(8, 44));

}  // namespace jxl

#endif  // ENCODER_ENTROPY_CODER_H_
