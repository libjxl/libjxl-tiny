// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#include "encoder/entropy_coder.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "encoder/ac_context.h"
#include "encoder/ac_strategy.h"
#include "encoder/base/bits.h"
#include "encoder/base/compiler_specific.h"
#include "encoder/base/profiler.h"
#include "encoder/base/status.h"
#include "encoder/coeff_order.h"
#include "encoder/coeff_order_fwd.h"
#include "encoder/common.h"
#include "encoder/dec_bit_reader.h"
#include "encoder/dec_context_map.h"
#include "encoder/fields.h"
#include "encoder/image.h"
#include "encoder/image_ops.h"

namespace jxl {

constexpr uint8_t BlockCtxMap::kDefaultCtxMap[];  // from ac_context.h

}  // namespace jxl
