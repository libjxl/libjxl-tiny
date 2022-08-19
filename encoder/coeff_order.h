// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_COEFF_ORDER_H_
#define ENCODER_COEFF_ORDER_H_

#include <stddef.h>
#include <stdint.h>

#include "encoder/ac_strategy.h"
#include "encoder/base/compiler_specific.h"
#include "encoder/base/status.h"
#include "encoder/coeff_order_fwd.h"
#include "encoder/common.h"
#include "encoder/dct_util.h"

namespace jxl {

// Those offsets get multiplied by kDCTBlockSize.
static constexpr size_t kCoeffOrderOffset[] = {
    0,    1,    2,    3,    4,    5,    6,    10,   14,   18,
    34,   50,   66,   68,   70,   72,   76,   80,   84,   92,
    100,  108,  172,  236,  300,  332,  364,  396,  652,  908,
    1164, 1292, 1420, 1548, 2572, 3596, 4620, 5132, 5644, 6156,
};
static_assert(3 * kNumOrders + 1 ==
                  sizeof(kCoeffOrderOffset) / sizeof(*kCoeffOrderOffset),
              "Update this array when adding or removing order types.");

static constexpr size_t CoeffOrderOffset(size_t order, size_t c) {
  return kCoeffOrderOffset[3 * order + c] * kDCTBlockSize;
}

static constexpr size_t kCoeffOrderMaxSize =
    kCoeffOrderOffset[3 * kNumOrders] * kDCTBlockSize;

// Mapping from AC strategy to order bucket. Strategies with different natural
// orders must have different buckets.
constexpr uint8_t kStrategyOrder[] = {
    0, 1, 1, 1, 2, 3, 4, 4, 5,  5,  6,  6,  1,  1,
    1, 1, 1, 1, 7, 8, 8, 9, 10, 10, 11, 12, 12,
};

static_assert(AcStrategy::kNumValidStrategies ==
                  sizeof(kStrategyOrder) / sizeof(*kStrategyOrder),
              "Update this array when adding or removing AC strategies.");

constexpr uint32_t kPermutationContexts = 8;

}  // namespace jxl

#endif  // ENCODER_COEFF_ORDER_H_
