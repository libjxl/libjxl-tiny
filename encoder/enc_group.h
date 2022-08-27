// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_ENC_GROUP_H_
#define ENCODER_ENC_GROUP_H_

#include <stddef.h>

#include "encoder/ac_strategy.h"
#include "encoder/base/status.h"
#include "encoder/chroma_from_luma.h"
#include "encoder/dct_util.h"
#include "encoder/image.h"
#include "encoder/quant_weights.h"

namespace jxl {

void ComputeCoefficients(size_t group_idx, const Image3F& opsin,
                         const ImageI& raw_quant_field,
                         const DequantMatrices& matrices, const float scale,
                         const ColorCorrelationMap& cmap,
                         const AcStrategyImage& ac_strategy,
                         const float x_qm_mul, ACImage* coeffs, Image3F* dc);

}  // namespace jxl

#endif  // ENCODER_ENC_GROUP_H_
