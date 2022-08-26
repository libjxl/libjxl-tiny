// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "encoder/ac_strategy.h"
#include "encoder/coeff_order.h"
#include "encoder/coeff_order_fwd.h"

namespace jxl {

void ComputeCoeffOrder(uint16_t used_acs, coeff_order_t* JXL_RESTRICT order) {
  std::vector<coeff_order_t> natural_order_buffer;
  uint16_t computed = 0;
  for (uint8_t o = 0; o < AcStrategy::kNumValidStrategies; ++o) {
    uint8_t ord = kStrategyOrder[o];
    if (computed & (1 << ord)) continue;
    computed |= 1 << ord;
    AcStrategy acs = AcStrategy::FromRawStrategy(o);
    size_t sz = kDCTBlockSize * acs.covered_blocks_x() * acs.covered_blocks_y();

    // Do nothing for transforms that don't appear.
    if ((1 << ord) & ~used_acs) continue;

    if (natural_order_buffer.size() < sz) natural_order_buffer.resize(sz);
    acs.ComputeNaturalCoeffOrder(natural_order_buffer.data());

    for (size_t c = 0; c < 3; c++) {
      size_t offset = CoeffOrderOffset(ord, c);
      JXL_DASSERT(CoeffOrderOffset(ord, c + 1) - offset == sz);
      memcpy(&order[offset], natural_order_buffer.data(), sz * sizeof(*order));
    }
  }
}

}  // namespace jxl
