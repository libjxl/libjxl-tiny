// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_ENC_COEFF_ORDER_H_
#define ENCODER_ENC_COEFF_ORDER_H_

#include <stddef.h>
#include <stdint.h>

#include "encoder/enc_bit_writer.h"
#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/aux_out_fwd.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/coeff_order.h"
#include "encoder/coeff_order_fwd.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dct_util.h"
#include "lib/jxl/dec_bit_reader.h"

namespace jxl {

// Orders that are actually used in part of image. `rect` is in block units.
// Returns {orders that are used, orders that might be made non-default}.
std::pair<uint32_t, uint32_t> ComputeUsedOrdersTiny(
    const AcStrategyImage& ac_strategy, const Rect& rect);

// Modify zig-zag order, so that DCT bands with more zeros go later.
// Order of DCT bands with same number of zeros is untouched, so
// permutation will be cheaper to encode.
void ComputeCoeffOrderTiny(const ACImage& acs,
                           const AcStrategyImage& ac_strategy,
                           const FrameDimensions& frame_dim,
                           uint32_t& used_orders, uint16_t used_acs,
                           coeff_order_t* JXL_RESTRICT order);

void EncodeCoeffOrders(uint16_t used_orders,
                       const coeff_order_t* JXL_RESTRICT order,
                       BitWriter* writer, size_t layer,
                       AuxOut* JXL_RESTRICT aux_out);

// Encoding/decoding of a single permutation. `size`: number of elements in the
// permutation. `skip`: number of elements to skip from the *beginning* of the
// permutation.
void EncodePermutation(const coeff_order_t* JXL_RESTRICT order, size_t skip,
                       size_t size, BitWriter* writer, int layer,
                       AuxOut* aux_out);

}  // namespace jxl

#endif  // ENCODER_ENC_COEFF_ORDER_H_
