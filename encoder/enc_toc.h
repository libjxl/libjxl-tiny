// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_ENC_TOC_H_
#define ENCODER_ENC_TOC_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "encoder/base/compiler_specific.h"
#include "encoder/base/status.h"
#include "encoder/coeff_order_fwd.h"
#include "encoder/enc_bit_writer.h"

namespace jxl {

// Writes the group offsets. If the permutation vector is nullptr, the identity
// permutation will be used.
Status WriteGroupOffsets(const std::vector<BitWriter>& group_codes,
                         const std::vector<coeff_order_t>* permutation,
                         BitWriter* JXL_RESTRICT writer);

}  // namespace jxl

#endif  // ENCODER_ENC_TOC_H_
