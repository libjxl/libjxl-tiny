// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_AC_CONTEXT_H_
#define ENCODER_AC_CONTEXT_H_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <vector>

#include "encoder/base/compiler_specific.h"

namespace jxl {

// The number of predicted nonzeros goes from 0 to 1008. We use
// ceil(log2(predicted+1)) as a context for the number of nonzeros, so from 0 to
// 10, inclusive.
constexpr uint32_t kNonZeroBuckets = 37;

static constexpr uint16_t kCoeffFreqContext[64] = {
    0xBAD, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
    15,    15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22,
    23,    23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26,
    27,    27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
};

static constexpr uint16_t kCoeffNumNonzeroContext[64] = {
    0xBAD, 0,   31,  62,  62,  93,  93,  93,  93,  123, 123, 123, 123,
    152,   152, 152, 152, 152, 152, 152, 152, 180, 180, 180, 180, 180,
    180,   180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206,
    206,   206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
    206,   206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
};

// Supremum of ZeroDensityContext(x, y) + 1, when x + y < 64.
constexpr int kZeroDensityContextCount = 458;
// Supremum of ZeroDensityContext(x, y) + 1.
constexpr int kZeroDensityContextLimit = 474;

static constexpr uint8_t kCompactBlockContextMap[] = {
    0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,  // Y
    2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3,  // X
    2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3,  // B
};
static constexpr uint8_t kBlockContextMap[] = {
    // X
    2, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0,  //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     //
    // Y
    0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,  //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     //
    // B
    2, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0,  //
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     //
};
static constexpr size_t kNumAcStrategyCodes = 27;
static constexpr size_t kNumBlockCtxs = 4;

static constexpr size_t BlockContext(size_t c, uint8_t ac_strategy_code) {
  return kBlockContextMap[c * kNumAcStrategyCodes + ac_strategy_code];
}

// Context map for AC coefficients consists of 2 blocks:
//  |num_ctxs x                : context for number of non-zeros in the block
//   kNonZeroBuckets|            computed from block context and predicted
//                               value (based top and left values)
//  |num_ctxs x                : context for AC coefficient symbols,
//   kZeroDensityContextCount|   computed from block context,
//                               number of non-zeros left and
//                               index in scan order
static constexpr size_t kNumACContexts =
    kNumBlockCtxs * (kNonZeroBuckets + kZeroDensityContextCount);

/* This function is used for entropy-sources pre-clustering.
 *
 * Ideally, each combination of |nonzeros_left| and |k| should go to its own
 * bucket; but it implies (64 * 63 / 2) == 2016 buckets. If there is other
 * dimension (e.g. block context), then number of primary clusters becomes too
 * big.
 *
 * To solve this problem, |nonzeros_left| and |k| values are clustered. It is
 * known that their sum is at most 64, consequently, the total number buckets
 * is at most A(64) * B(64).
 */
static JXL_INLINE size_t ZeroDensityContext(size_t nonzeros_left, size_t k,
                                            size_t covered_blocks,
                                            size_t log2_covered_blocks,
                                            size_t prev) {
  nonzeros_left = (nonzeros_left + covered_blocks - 1) >> log2_covered_blocks;
  k >>= log2_covered_blocks;
  return (kCoeffNumNonzeroContext[nonzeros_left] + kCoeffFreqContext[k]) * 2 +
         prev;
}

static constexpr uint32_t ZeroDensityContextsOffset(uint32_t block_ctx) {
  return (kNumBlockCtxs * kNonZeroBuckets +
          kZeroDensityContextCount * block_ctx);
}

// Non-zero context is based on number of non-zeros and block context.
// For better clustering, contexts with same number of non-zeros are grouped.
static constexpr uint32_t NonZeroContext(uint32_t non_zeros,
                                         uint32_t block_ctx) {
  return (non_zeros < 8     ? non_zeros
          : non_zeros >= 64 ? 36
                            : 4 + non_zeros / 2) *
             kNumBlockCtxs +
         block_ctx;
}

}  // namespace jxl

#endif  // ENCODER_AC_CONTEXT_H_
