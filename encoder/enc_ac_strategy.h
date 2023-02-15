// Copyright (c) the JPEG XL Project Authors.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file or at
// https://developers.google.com/open-source/licenses/bsd

#ifndef ENCODER_ENC_AC_STRATEGY_H_
#define ENCODER_ENC_AC_STRATEGY_H_

#include "encoder/ac_strategy.h"
#include "encoder/base/status.h"
#include "encoder/image.h"
#include "encoder/quant_weights.h"

namespace jxl {

void FindBest16x16Transform(const Image3F& opsin, const Rect& block_rect,
                            size_t bx, size_t by, size_t cx, size_t cy,
                            float distance, const DequantMatrices& matrices,
                            const ImageF& qf, const ImageF& maskf, int8_t ytox,
                            int8_t ytob,
                            AcStrategyImage* JXL_RESTRICT ac_strategy,
                            float* block, float* scratch_space);

void AdjustQuantField(const AcStrategyImage& ac_strategy,
                      const Rect& block_rect, ImageB* quant_field);

}  // namespace jxl

#endif  // ENCODER_ENC_AC_STRATEGY_H_
