// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_ENC_GROUP_H_
#define ENCODER_ENC_GROUP_H_

#include <stddef.h>
#include <stdint.h>

#include "encoder/enc_cache.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_bit_writer.h"

namespace jxl {

// Fills DC
void ComputeCoefficientsTiny(size_t group_idx, PassesEncoderState* enc_state,
                             const Image3F& opsin, Image3F* dc);

Status EncodeGroupTokenizedCoefficients(size_t group_idx, size_t pass_idx,
                                        size_t histogram_idx,
                                        const PassesEncoderState& enc_state,
                                        BitWriter* writer);

}  // namespace jxl

#endif  // ENCODER_ENC_GROUP_H_
