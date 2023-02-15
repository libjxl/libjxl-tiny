// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_ENC_GROUP_H_
#define ENCODER_ENC_GROUP_H_

#include <stddef.h>

#include "encoder/ac_strategy.h"
#include "encoder/dc_group_data.h"
#include "encoder/enc_bit_writer.h"
#include "encoder/entropy_code.h"
#include "encoder/image.h"
#include "encoder/quant_weights.h"

namespace jxl {

void WriteACGroup(const Image3F& opsin, const Rect& group_brect,
                  const DequantMatrices& matrices, const float scale,
                  const float scale_dc, const uint32_t x_qm_scale,
                  DCGroupData* dc_data, const EntropyCode& ac_code,
                  BitWriter* writer);

}  // namespace jxl

#endif  // ENCODER_ENC_GROUP_H_
