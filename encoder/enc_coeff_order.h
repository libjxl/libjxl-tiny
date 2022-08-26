// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_ENC_COEFF_ORDER_H_
#define ENCODER_ENC_COEFF_ORDER_H_

#include <stddef.h>
#include <stdint.h>

#include "encoder/base/compiler_specific.h"
#include "encoder/base/status.h"
#include "encoder/coeff_order.h"
#include "encoder/coeff_order_fwd.h"

namespace jxl {

// Fill in zig-zag order for each DCT block size.
void ComputeCoeffOrder(uint16_t used_acs, coeff_order_t* JXL_RESTRICT order);

}  // namespace jxl

#endif  // ENCODER_ENC_COEFF_ORDER_H_
